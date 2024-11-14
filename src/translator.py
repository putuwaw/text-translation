from typing import Optional
from langdetect import detect


def translate(text: Optional[str]) -> Optional[str]:
    if text:
        # TODO: add model to translate here
        return text.upper()
    return None

def verify_lang(text: str, comparison: str) -> bool:
    return detect(text) == comparison