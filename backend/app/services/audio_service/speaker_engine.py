import sherpa_onnx
import numpy as np
from backend.utils.logger import log

class SpeakerEngine:
    def __init__(self, model_path, num_threads=2):
        config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=model_path, num_threads=num_threads, provider="cpu"
        )
        self.extractor = sherpa_onnx.SpeakerEmbeddingExtractor(config)
        self.manager = sherpa_onnx.SpeakerEmbeddingManager(self.extractor.dim)
        self.speaker_count = 0
        log.info(f"SpeakerEngine initialized with model {model_path}")        

    def identify(self, samples, threshold=0.6):
        """识别或注册说话人"""
        stream = self.extractor.create_stream()
        stream.accept_waveform(sample_rate=16000, waveform=samples)
        stream.input_finished()
        
        embedding = self.extractor.compute(stream)
        name = self.manager.search(embedding, threshold=threshold)
        
        if not name:
            name = f"Speaker_{self.speaker_count}"
            self.manager.add(name, embedding)
            self.speaker_count += 1
        return name