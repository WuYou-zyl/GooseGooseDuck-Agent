from __future__ import annotations

import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import sherpa_onnx

from backend.app.core.ggd_coordinator import GGDCoordinator
from backend.app.services.audio_service.asr_engine import ASREngine
from backend.app.services.audio_service.audio_recorder import AudioRecorder
from backend.app.services.audio_service.speaker_engine import SpeakerEngine
from backend.utils.config_loader import config
from backend.utils.logger import log
from backend.utils.pathtool import get_abs_path


class AudioCaptureService:
    """Optimized loopback audio pipeline using Sherpa ONNX VAD/ASR.

    The recorder only captures audio. VAD cuts speech segments, then ASR and
    speaker embedding recognition run in parallel executors so capture does not
    block on model inference.
    """

    def __init__(self, coordinator: GGDCoordinator):
        self.recorder = AudioRecorder()
        model_base = get_abs_path("backend/models")

        self.speaker_engine = SpeakerEngine(str(model_base / config.audio.spk_model))
        asr_dir = model_base / config.asr.asr_dir
        config.asr.tokens = str(asr_dir / config.asr.tokens)
        config.asr.asr_model = str(asr_dir / config.asr.asr_model)
        config.asr.graph = str(asr_dir / config.asr.graph)
        config.asr.encoder = str(asr_dir / config.asr.encoder)
        config.asr.decoder = str(asr_dir / config.asr.decoder)
        config.asr.joiner = str(asr_dir / config.asr.joiner)
        self.asr_engine = ASREngine(args=config.asr)
        self.coordinator = coordinator

        vad_config = sherpa_onnx.VadModelConfig()
        vad_config.silero_vad.model = str(model_base / config.audio.vad_model)
        vad_config.silero_vad.threshold = float(config.audio.vad_threshold)
        vad_config.silero_vad.min_silence_duration = float(config.audio.min_silence_time)
        vad_config.silero_vad.min_speech_duration = float(config.audio.min_segment_time)
        vad_config.silero_vad.max_speech_duration = float(config.audio.max_segment_time)
        vad_config.sample_rate = int(config.audio.sample_rate)
        self.vad = sherpa_onnx.VoiceActivityDetector(
            vad_config,
            buffer_size_in_seconds=30,
        )
        log.info(f"VAD initialized with model: {vad_config.silero_vad.model}")

        self.audio_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=100)
        self.running = False
        self.stream_start_wall_time = 0.0
        self.worker_thread: threading.Thread | None = None

        self.segment_executor = ThreadPoolExecutor(max_workers=2)
        self.asr_executor = ThreadPoolExecutor(max_workers=1)
        self.speaker_executor = ThreadPoolExecutor(max_workers=1)
        self.segment_semaphore = threading.Semaphore(4)
        self.result_lock = threading.Lock()

    def _process_segment_async(
        self,
        samples: np.ndarray,
        precise_start_time: float,
        duration: float,
    ) -> None:
        try:
            asr_future = self.asr_executor.submit(self.asr_engine.transcribe, samples)
            speaker_future = self.speaker_executor.submit(
                self.speaker_engine.identify,
                samples,
            )

            text = asr_future.result()
            speaker_name = speaker_future.result()
            log.debug(
                "audio speaker=%s text=%s start=%s duration=%s",
                speaker_name,
                text,
                precise_start_time,
                duration,
            )

            if self.coordinator and text:
                with self.result_lock:
                    self.coordinator.on_audio_result(
                        text,
                        precise_start_time,
                        duration,
                        speaker_name,
                    )
        except Exception as e:
            log.exception(f"Audio segment processing failed: {e}")
        finally:
            self.segment_semaphore.release()

    def _processing_worker(self) -> None:
        while self.running:
            try:
                samples = self.audio_queue.get(timeout=1)
            except queue.Empty:
                continue

            try:
                self.vad.accept_waveform(samples)
                while not self.vad.empty():
                    segment = self.vad.front
                    if len(segment.samples) < config.audio.min_segment_time * config.audio.sample_rate:
                        self.vad.pop()
                        continue

                    precise_start_time = (
                        self.stream_start_wall_time
                        + (segment.start / float(config.audio.sample_rate))
                    )
                    duration = len(segment.samples) / float(config.audio.sample_rate)
                    segment_samples = np.copy(segment.samples)

                    if self.segment_semaphore.acquire(blocking=False):
                        self.segment_executor.submit(
                            self._process_segment_async,
                            segment_samples,
                            precise_start_time,
                            duration,
                        )
                    else:
                        log.warning("Dropping audio segment because processing is backlogged")

                    self.vad.pop()
            finally:
                self.audio_queue.task_done()

    def run(self) -> None:
        if self.running:
            return
        self.running = True
        self.stream_start_wall_time = time.time()
        log.info("Optimized system audio capture started.")

        self.worker_thread = threading.Thread(
            target=self._processing_worker,
            name="OptimizedAudioProcessingWorker",
            daemon=True,
        )
        self.worker_thread.start()

        try:
            for samples, _timestamp in self.recorder.record_generator():
                if not self.running:
                    break
                try:
                    self.audio_queue.put_nowait(samples)
                except queue.Full:
                    log.warning("Audio processing queue is full; dropping captured chunk")
        except Exception as e:
            log.error(f"Audio capture failed: {e}")
        finally:
            self.running = False

    def stop(self) -> None:
        self.running = False
        self.recorder.stop()
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=3)
        self.segment_executor.shutdown(wait=False, cancel_futures=True)
        self.asr_executor.shutdown(wait=False, cancel_futures=True)
        self.speaker_executor.shutdown(wait=False, cancel_futures=True)
