"""Stand-in for ``DashAI.back.core.utils``."""


class MultilingualString:
    """Holds one text per supported language."""

    def __init__(
        self, en: str = "", es: str = "", pt: str = "", de: str = "", zh: str = ""
    ):
        self.en, self.es, self.pt, self.de, self.zh = en, es, pt, de, zh

    def __str__(self) -> str:
        return self.en
