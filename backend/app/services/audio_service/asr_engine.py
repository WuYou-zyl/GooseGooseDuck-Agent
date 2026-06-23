import sherpa_onnx
from backend.utils.logger import log

class ASREngine:
    def __init__(self, args, num_threads=4):
        # 使用 Zipformer CTC + HLG 架构
        self.recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=args.tokens,
            encoder=args.encoder,
            decoder=args.decoder,
            joiner=args.joiner,
            num_threads=4,
            sample_rate=16000,
            feature_dim=80,
            decoding_method="greedy_search",
            provider=args.provider,
        )
        log.info("ASR engine initialized")

    def transcribe(self, samples):
        stream = self.recognizer.create_stream()
        stream.accept_waveform(16000, samples)
        # 加上尾部静音以强制刷新识别结果[cite: 3]
        import numpy as np
        stream.accept_waveform(16000, np.zeros(int(0.3 * 16000), dtype=np.float32))
        stream.input_finished()

        while self.recognizer.is_ready(stream):
            self.recognizer.decode_stream(stream)
        return self.recognizer.get_result(stream)