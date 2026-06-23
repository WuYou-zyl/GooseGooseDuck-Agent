import pyaudio
import wave
import numpy as np
import time
import threading
import json
import os

from backend.utils.asr_hotwords import load_hotwords_text, merge_hotwords_text

# FunASR 导入
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess


class GooseGooseDuckAudioAnalyzer:
    """
    从 `GGD-AI-main/extract_speaker_statement.py` 复制而来，作为 legacy 复用。
    A 侧新代码请通过 backend/services/asr_service.py 调用。
    """

    def __init__(self, on_new_record=None, auto_save=True, preloaded_model=None, dynamic_hotwords_provider=None):
        # 音频参数
        self.format = pyaudio.paInt16
        self.channels = 2
        self.rate = 44100
        self.chunk = 1024

        # 使用VB-Cable Output作为输入设备
        self.device_index = self._find_vbcable_device()

        if preloaded_model is not None:
            self.funasr_model = preloaded_model
        else:
            import torch
            _device = "cuda:0" if torch.cuda.is_available() else "cpu"
            print(f"[ASR] Loading SenseVoice-Small on {_device}", flush=True)
            self.funasr_model = AutoModel(
                model="iic/SenseVoiceSmall",
                vad_model="fsmn-vad",
                vad_kwargs={"max_single_segment_time": 30000},
                trust_remote_code=True,
                device=_device,
            )

        self.is_recording = False

        self.conversation_log = []
        self._log_lock = threading.Lock()

        self._current_speaker = "unknown"
        self._speaker_lock = threading.Lock()

        self._audio_buffer = []
        self._buffer_lock = threading.Lock()

        self.on_new_record = on_new_record
        self.auto_save = auto_save
        self.hotword_text = load_hotwords_text()
        self.dynamic_hotwords_provider = dynamic_hotwords_provider

        self._thread = None
        self._pyaudio = None
        self._stream = None

    def _get_hotword_text(self):
        provider = self.dynamic_hotwords_provider
        if provider is None:
            return self.hotword_text
        try:
            extra_words = provider() or []
        except Exception:
            extra_words = []
        return merge_hotwords_text(self.hotword_text, extra_words)

    def _find_vbcable_device(self):
        p = pyaudio.PyAudio()
        for i in range(p.get_device_count()):
            dev = p.get_device_info_by_index(i)
            if "CABLE Output" in dev["name"] and dev["hostApi"] == 0:
                return i
        raise Exception("未找到VB-Cable Output设备，请先安装虚拟声卡")

    def start(self):
        if self.is_recording:
            return
        self.is_recording = True
        self._thread = threading.Thread(target=self._record_loop, daemon=True)
        self._thread.start()

    def stop(self, round_num: int = 1):
        if not self.is_recording:
            return
        self.is_recording = False
        try:
            if self._thread and self._thread.is_alive():
                self._thread.join(timeout=2.0)
        except Exception:
            pass

        self._flush_buffer(speaker_override=self._current_speaker, round_num=round_num)

        try:
            if self._stream is not None:
                try:
                    self._stream.stop_stream()
                except Exception:
                    pass
                try:
                    self._stream.close()
                except Exception:
                    pass
        finally:
            self._stream = None
        try:
            if self._pyaudio is not None:
                self._pyaudio.terminate()
        finally:
            self._pyaudio = None

    def _record_loop(self):
        self._pyaudio = pyaudio.PyAudio()
        self._stream = self._pyaudio.open(
            format=self.format,
            channels=self.channels,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk,
            input_device_index=self.device_index,
        )
        print(f"[AUDIO] Recording started on device {self.device_index}", flush=True)

        chunk_sec = self.chunk / self.rate

        # --- Tunable parameters ---
        voice_threshold = 200          # lowered from 500; game audio via VB-Cable is quieter
        speech_start_frames = 3        # need 3 consecutive voice frames to enter speech state
        silence_flush_sec = 2.5        # seconds of silence before flushing segment
        max_segment_sec = 20.0         # force-flush after this many seconds
        min_flush_sec = 0.8            # don't bother ASR with segments shorter than this

        silence_flush_frames = int(silence_flush_sec / chunk_sec)
        max_segment_frames = int(max_segment_sec / chunk_sec)

        silent_count = 0
        voice_streak = 0               # consecutive voice frames (for speech start hysteresis)
        segment_frames = 0             # total frames (voice + silence) in current segment
        is_speaking = False
        total_frames = 0

        try:
            while self.is_recording:
                try:
                    data = self._stream.read(self.chunk, exception_on_overflow=False)
                except Exception:
                    time.sleep(0.01)
                    continue

                total_frames += 1
                volume = self._frame_volume(data)
                has_voice = volume > voice_threshold

                if has_voice:
                    voice_streak += 1
                    silent_count = 0
                else:
                    voice_streak = 0

                if not is_speaking:
                    if voice_streak >= speech_start_frames:
                        is_speaking = True
                        segment_frames = voice_streak
                        print(f"[AUDIO] Speech started (speaker={self._current_speaker}, vol={volume:.0f})", flush=True)
                        # retroactively buffer the streak frames we missed
                        # (current frame is the Nth voice frame; previous N-1 are lost,
                        #  but that's only ~70ms — acceptable)
                        with self._buffer_lock:
                            self._audio_buffer.append(data)
                else:
                    # In speaking state: buffer EVERYTHING (voice + silence)
                    with self._buffer_lock:
                        self._audio_buffer.append(data)
                    segment_frames += 1

                    if has_voice:
                        silent_count = 0
                    else:
                        silent_count += 1

                    # Flush on prolonged silence
                    if silent_count >= silence_flush_frames:
                        seg_dur = segment_frames * chunk_sec
                        print(f"[AUDIO] Silence detected ({silence_flush_sec}s), segment={seg_dur:.1f}s", flush=True)
                        is_speaking = False
                        segment_frames = 0
                        silent_count = 0
                        voice_streak = 0
                        self._flush_buffer(min_duration=min_flush_sec)

                    # Force flush on max segment length
                    elif segment_frames >= max_segment_frames:
                        seg_dur = segment_frames * chunk_sec
                        print(f"[AUDIO] Max segment {seg_dur:.1f}s reached, flushing", flush=True)
                        self._flush_buffer(min_duration=min_flush_sec)
                        segment_frames = 0
                        silent_count = 0
                        # stay in speaking state

                if total_frames % 2000 == 0:
                    buf_dur = len(self._audio_buffer) * chunk_sec
                    print(f"[AUDIO] stats: total={total_frames}, buf={buf_dur:.1f}s, speaking={is_speaking}, speaker={self._current_speaker}", flush=True)
        finally:
            try:
                if self._stream is not None:
                    self._stream.stop_stream()
            except Exception:
                pass

    def _flush_buffer(self, speaker_override: str = None, round_num: int = 1, min_duration: float = 0.0):
        """Flush the audio buffer and send to transcription.
        speaker_override: pass explicitly to avoid re-acquiring _speaker_lock.
        min_duration: discard segments shorter than this (seconds).
        """
        buffer_copy = None
        speaker = speaker_override if speaker_override else self._current_speaker
        with self._buffer_lock:
            if self._audio_buffer:
                buffer_copy = self._audio_buffer.copy()
                self._audio_buffer = []
        if buffer_copy:
            duration = len(buffer_copy) * self.chunk / self.rate
            if duration < min_duration:
                print(f"[AUDIO] Discarding short segment ({duration:.1f}s < {min_duration}s)", flush=True)
                return
            print(f"[AUDIO] Flushing {len(buffer_copy)} frames ({duration:.1f}s) for speaker {speaker}", flush=True)
            threading.Thread(
                target=self._process_speech,
                args=(buffer_copy, speaker, round_num),
                daemon=True,
            ).start()

    def set_speaker(self, speaker, round_num: int = 1):
        with self._speaker_lock:
            if speaker != self._current_speaker:
                old_speaker = self._current_speaker
                print(f"[AUDIO] Speaker change: {old_speaker} -> {speaker}", flush=True)
                self._flush_buffer(speaker_override=old_speaker, round_num=round_num)
                self._current_speaker = speaker

    def get_speaker(self):
        with self._speaker_lock:
            return self._current_speaker

    def _frame_volume(self, audio_data):
        """Return mean absolute amplitude of one audio chunk."""
        audio_np = np.frombuffer(audio_data, dtype=np.int16)
        return np.abs(audio_np).mean()

    def transcribe_audio(self, audio_frames):
        """Transcribe audio using SenseVoice. Returns (clean_text, raw_text)."""
        if not audio_frames:
            return "", ""

        audio_data = b"".join(audio_frames)

        temp_dir = "test_audio"
        os.makedirs(temp_dir, exist_ok=True)
        temp_file = os.path.join(temp_dir, f"temp_audio_{int(time.time() * 1000)}.wav")

        with wave.open(temp_file, "wb") as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(2)
            wf.setframerate(self.rate)
            wf.writeframes(audio_data)

        try:
            result = self.funasr_model.generate(
                input=temp_file,
                cache={},
                language="auto",
                hotword=self._get_hotword_text(),
                use_itn=True,
                batch_size_s=60,
                merge_vad=True,
                merge_length_s=15,
            )
            if result and len(result) > 0:
                raw_text = result[0].get("text", "").strip()
                clean_text = rich_transcription_postprocess(raw_text)
                return clean_text, raw_text
            return "", ""
        finally:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except Exception:
                pass

    def analyze_emotion(self, raw_text):
        """Extract emotion from SenseVoice's built-in tags like <|HAPPY|>, <|SAD|>, <|ANGRY|>."""
        import re
        emotion_map = {
            "HAPPY": "positive",
            "SAD": "negative",
            "ANGRY": "negative",
            "SURPRISED": "neutral",
            "FEARFUL": "negative",
            "DISGUSTED": "negative",
            "NEUTRAL": "neutral",
        }
        tags = re.findall(r"<\|(\w+)\|>", raw_text)
        for tag in tags:
            if tag in emotion_map:
                return emotion_map[tag]
        return "neutral"

    def _process_speech(self, audio_frames, speaker, round_num: int = 1):
        duration = len(audio_frames) * self.chunk / self.rate
        if duration < 0.3:
            return
        print(f"[ASR] Transcribing {duration:.1f}s audio for speaker {speaker} ...", flush=True)
        clean_text, raw_text = self.transcribe_audio(audio_frames)
        if clean_text:
            emotion = self.analyze_emotion(raw_text)
            timestamp = time.strftime("%H:%M:%S")
            record = {
                "timestamp": timestamp,
                "text": clean_text,
                "emotion": emotion,
                "speaker": speaker,
                "duration": round(duration, 2),
                "round": round_num,
            }
            print(f"[ASR] Result: [{speaker}] {clean_text} (emotion={emotion})", flush=True)
            with self._log_lock:
                self.conversation_log.append(record)
            if self.on_new_record:
                self.on_new_record(record)
            if self.auto_save:
                self._save_to_file("game_analysis.json")
        else:
            print(f"[ASR] No text recognized for {duration:.1f}s audio", flush=True)

    def _save_to_file(self, filename):
        path = filename
        if filename == "game_analysis.json":
            try:
                from backend.utils.path_tool import get_abs_path

                path = get_abs_path(os.path.join("data", "game_analysis.json"))
            except Exception:
                path = filename
        with self._log_lock:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.conversation_log, f, ensure_ascii=False, indent=2)
