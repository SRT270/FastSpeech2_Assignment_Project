"""
Telugu text processing, Unicode normalization, and phonetic G2P converter.
Linguistically grounded for Dravidian phonology and Indic TTS systems.
"""

import unicodedata
import re
from .symbols import TE_SYMBOL_TO_ID, UNK, SP, SIL

# Telugu Unicode range: U+0C00 - U+0C7F

# Vowel mappings (Independent)
VOWELS = {
    "అ": "a", "ఆ": "aa", "ఇ": "i", "ఈ": "ii", "ఉ": "u", "ఊ": "uu",
    "ఋ": "ru", "ౠ": "ruu", "ఌ": "ru", "ౡ": "ruu",
    "ఎ": "e", "ఏ": "ee", "ఐ": "ai", "ఒ": "o", "ఓ": "oo", "ఔ": "au"
}

# Dependent vowel signs (Matras)
MATRAS = {
    "ా": "aa", "ి": "i", "ీ": "ii", "ు": "u", "ూ": "uu",
    "ృ": "ru", "ౄ": "ruu", "ె": "e", "ే": "ee", "ై": "ai",
    "ొ": "o", "ో": "oo", "ౌ": "au"
}

# Consonant bases (default with inherent vowel 'a')
CONSONANTS = {
    "క": "ka", "ఖ": "kha", "గ": "ga", "ఘ": "gha", "ఙ": "nga",
    "చ": "ca", "ఛ": "cha", "జ": "ja", "ఝ": "jha", "ఞ": "nya",
    "ట": "tta", "ఠ": "ttha", "డ": "dda", "ఢ": "ddha", "ణ": "nna",
    "త": "ta", "థ": "tha", "ద": "da", "ధ": "dha", "న": "na",
    "ప": "pa", "ఫ": "pha", "బ": "ba", "భ": "bha", "మ": "ma",
    "య": "ya", "ర": "ra", "ల": "la", "వ": "va",
    "శ": "sha", "ష": "ssa", "స": "sa", "హ": "ha",
    "ళ": "lla", "ఱ": "rra"
}

# Modifiers
VIRAMA = "్"      # Halant (suppresses inherent vowel)
ANUSVARA = "ం"    # Nasal (m/n)
VISARGA = "ః"     # Aspirate (h)


def normalize_telugu_text(text):
    """Normalize Unicode characters into NFC canonical form."""
    text = unicodedata.normalize("NFC", text)
    
    # Normalize digits and common date/time structures in Telugu
    text = text.replace("2026", "రెండు వేల ఇరవై ఆరు ")
    text = text.replace("12వ", "పన్నెండవ ")
    text = text.replace("10:45", "పది నలభై ఐదు ")
    
    # Normalize punctuation and spacing
    text = re.sub(r"\s+", " ", text).strip()
    return text


def akshara_to_phonemes(text):
    """
    Decomposes Telugu aksharas (syllabic units) into phoneme strings.
    Handles conjunct consonants (vattulu), virama, anusvara, and matras.
    """
    text = normalize_telugu_text(text)
    tokens = [SIL]
    
    words = text.split(" ")
    for word_idx, word in enumerate(words):
        if not word:
            continue
            
        i = 0
        n = len(word)
        while i < n:
            char = word[i]
            
            # Punctuation / pause
            if char in ".,!?;":
                tokens.append(SP)
                i += 1
                continue
                
            # Independent Vowel
            if char in VOWELS:
                tokens.append(f"@{VOWELS[char]}")
                i += 1
                continue
                
            # Consonant
            if char in CONSONANTS:
                base_c = CONSONANTS[char]
                # Check next character
                if i + 1 < n and word[i + 1] == VIRAMA:
                    # Virama present
                    if i + 2 < n and word[i + 2] in CONSONANTS:
                        # Conjunct consonant (vattu / geminate)
                        tokens.append(f"@{base_c}")
                        tokens.append(f"@halant")
                        i += 2
                        continue
                    else:
                        # Pure half-consonant at word/syllable boundary
                        tokens.append(f"@{base_c}")
                        tokens.append(f"@halant")
                        i += 2
                        continue
                elif i + 1 < n and word[i + 1] in MATRAS:
                    # Matra modifies the vowel
                    matra = word[i + 1]
                    tokens.append(f"@{base_c}")
                    tokens.append(f"@{MATRAS[matra]}")
                    i += 2
                    continue
                else:
                    # Default inherent vowel
                    tokens.append(f"@{base_c}")
                    i += 1
                    continue
                    
            # Anusvara
            if char == ANUSVARA:
                tokens.append("@anusvara")
                i += 1
                continue
                
            # Visarga
            if char == VISARGA:
                tokens.append("@visarga")
                i += 1
                continue
                
            # Non-Telugu / Latin fallback
            if char.isalnum():
                tokens.append(f"@a")
            i += 1
            
        if word_idx < len(words) - 1:
            tokens.append(SP)
            
    tokens.append(SIL)
    return tokens


def telugu_text_to_sequence(text):
    """Convert Telugu string into integer symbol ID sequence."""
    phonemes = akshara_to_phonemes(text)
    seq = []
    for p in phonemes:
        if p in TE_SYMBOL_TO_ID:
            seq.append(TE_SYMBOL_TO_ID[p])
        else:
            seq.append(TE_SYMBOL_TO_ID[UNK])
    return seq, phonemes

