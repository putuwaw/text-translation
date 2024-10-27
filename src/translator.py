from typing import Optional


def translate(text: Optional[str]) -> Optional[str]:
    if text:
        # TODO: add model to translate here
        return text.upper()
    return None
