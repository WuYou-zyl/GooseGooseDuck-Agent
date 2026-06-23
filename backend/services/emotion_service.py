from __future__ import annotations


class SimpleEmotionService:
    """Cheap heuristic emotion summary from transcript text."""

    def infer(self, text: str) -> str:
        negative_words = ["杀", "狼", "怀疑", "假的", "骗", "刀"]
        positive_words = ["信", "好", "帮", "保", "村民"]
        score = sum(1 for w in negative_words if w in text) - sum(1 for w in positive_words if w in text)
        return "语气可疑/指控" if score > 0 else "语气可信/辩护" if score < 0 else "语气中性"

