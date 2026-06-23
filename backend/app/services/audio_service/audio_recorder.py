import time
import numpy as np
import soundcard as sc
from backend.utils.logger import log

class AudioRecorder:
    def __init__(self, sample_rate=16000, block_size=1600):
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.is_running = False

    def _get_loopback_device(self):
        """锁定系统的回环采样设备（扬声器输出）[cite: 6]"""
        try:
            default_speaker = sc.default_speaker()
            mics = sc.all_microphones(include_loopback=True)
            for mic in mics:
                if getattr(mic, "isloopback", False) and mic.id == default_speaker.id:
                    log.success(f"已锁定回环设备: {mic.name}")
                    return mic
            return sc.default_microphone()
        except Exception as e:
            log.error(f"探测音频设备失败: {e}")
            return None

    def record_generator(self):
        device = self._get_loopback_device()
        if not device: return

        self.is_running = True
        with device.recorder(samplerate=self.sample_rate, channels=1) as recorder:
            while self.is_running:
                data = recorder.record(numframes=self.block_size)
                # 预处理：混音与转换，合并双声道
                if data.ndim > 1: data = data.mean(axis=1)
                yield data.astype(np.float32), time.time()

    def stop(self):
        self.is_running = False