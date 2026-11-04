from .symbols import (
    ENGLISH_SYMBOLS,
    TELUGU_SYMBOLS,
    EN_SYMBOL_TO_ID,
    EN_ID_TO_SYMBOL,
    TE_SYMBOL_TO_ID,
    TE_ID_TO_SYMBOL,
    get_symbols,
)
from .english import english_to_phonemes, english_text_to_sequence
from .telugu import akshara_to_phonemes, telugu_text_to_sequence


def text_to_sequence(text, language="English"):
    if language.lower() in ["english", "en"]:
        return english_text_to_sequence(text)
    elif language.lower() in ["telugu", "te"]:
        return telugu_text_to_sequence(text)
    else:
        raise ValueError(f"Unsupported language: {language}")


__all__ = [
    "ENGLISH_SYMBOLS",
    "TELUGU_SYMBOLS",
    "EN_SYMBOL_TO_ID",
    "EN_ID_TO_SYMBOL",
    "TE_SYMBOL_TO_ID",
    "TE_ID_TO_SYMBOL",
    "get_symbols",
    "english_to_phonemes",
    "english_text_to_sequence",
    "akshara_to_phonemes",
    "telugu_text_to_sequence",
    "text_to_sequence",
]

