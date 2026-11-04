"""
Phoneme symbol inventories for English and Telugu FastSpeech 2.
"""

PAD = "_"
SIL = "@sil"
SP = "@sp"
UNK = "@unk"

SPECIAL_SYMBOLS = [PAD, SIL, SP, UNK]
PUNCTUATION = list("!'(),-.:;? ")

# English ARPAbet phonemes
ENGLISH_ARPABET = [
    "AA", "AA0", "AA1", "AA2", "AE", "AE0", "AE1", "AE2", "AH", "AH0", "AH1", "AH2",
    "AO", "AO0", "AO1", "AO2", "AW", "AW0", "AW1", "AW2", "AY", "AY0", "AY1", "AY2",
    "B", "CH", "D", "DH", "EH", "EH0", "EH1", "EH2", "ER", "ER0", "ER1", "ER2",
    "EY", "EY0", "EY1", "EY2", "F", "G", "HH", "IH", "IH0", "IH1", "IH2",
    "IY", "IY0", "IY1", "IY2", "JH", "K", "L", "M", "N", "NG", "OW", "OW0", "OW1", "OW2",
    "OY", "OY0", "OY1", "OY2", "P", "R", "S", "SH", "T", "TH", "UH", "UH0", "UH1", "UH2",
    "UW", "UW0", "UW1", "UW2", "V", "W", "Y", "Z", "ZH"
]

# Telugu Phonetic Inventory (Akshara / IPA-derived)
TELUGU_PHONEMES = [
    # Vowels
    "a", "aa", "i", "ii", "u", "uu", "ru", "ruu", "e", "ee", "ai", "o", "oo", "au",
    # Velar consonants
    "ka", "kha", "ga", "gha", "nga",
    # Palatal consonants
    "ca", "cha", "ja", "jha", "nya",
    # Retroflex consonants
    "tta", "ttha", "dda", "ddha", "nna",
    # Dental consonants
    "ta", "tha", "da", "dha", "na",
    # Labial consonants
    "pa", "pha", "ba", "bha", "ma",
    # Semi-vowels, sibilants, liquids
    "ya", "ra", "la", "va", "sha", "ssa", "sa", "ha", "lla", "rra", "ksha",
    # Anusvara & Visarga
    "anusvara", "visarga", "halant"
]

# Complete symbol lists per language
ENGLISH_SYMBOLS = SPECIAL_SYMBOLS + PUNCTUATION + [f"@{p}" for p in ENGLISH_ARPABET]
TELUGU_SYMBOLS = SPECIAL_SYMBOLS + PUNCTUATION + [f"@{p}" for p in TELUGU_PHONEMES]

# Unified vocabulary mapping
ALL_SYMBOLS = SPECIAL_SYMBOLS + PUNCTUATION + [f"@{p}" for p in ENGLISH_ARPABET] + [f"@te_{p}" for p in TELUGU_PHONEMES]

EN_SYMBOL_TO_ID = {s: i for i, s in enumerate(ENGLISH_SYMBOLS)}
EN_ID_TO_SYMBOL = {i: s for i, s in enumerate(ENGLISH_SYMBOLS)}

TE_SYMBOL_TO_ID = {s: i for i, s in enumerate(TELUGU_SYMBOLS)}
TE_ID_TO_SYMBOL = {i: s for i, s in enumerate(TELUGU_SYMBOLS)}


def get_symbols(language="English"):
    if language.lower() in ["english", "en"]:
        return ENGLISH_SYMBOLS, EN_SYMBOL_TO_ID, EN_ID_TO_SYMBOL
    elif language.lower() in ["telugu", "te"]:
        return TELUGU_SYMBOLS, TE_SYMBOL_TO_ID, TE_ID_TO_SYMBOL
    else:
        raise ValueError(f"Unsupported language: {language}")

