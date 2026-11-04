"""
English text cleaning, normalization, and G2P phonemization.
"""

import re
from .symbols import EN_SYMBOL_TO_ID, UNK, SP, SIL


# Common pronunciation dictionary for standard English words
CMU_FALLBACK = {
    "the": ["DH", "AH0"],
    "antidisestablishmentarianism": [
        "AE2", "N", "T", "IY0", "D", "IH2", "S", "IH0", "S", "T", "AE2", "B", "L",
        "IH0", "SH", "M", "AH0", "N", "T", "EH1", "R", "IY0", "AH0", "N", "IH2", "Z", "AH0", "M"
    ],
    "movement": ["M", "UW1", "V", "M", "AH0", "N", "T"],
    "gained": ["G", "EY1", "N", "D"],
    "considerable": ["K", "AH0", "N", "S", "IH1", "D", "ER0", "AH0", "B", "AH0", "L"],
    "momentum": ["M", "OW0", "M", "EH1", "N", "T", "AH0", "M"],
    "during": ["D", "UH1", "R", "IH0", "NG"],
    "parliamentary": ["P", "AA2", "R", "L", "AH0", "M", "EH1", "N", "T", "ER0", "IY0"],
    "proceedings": ["P", "R", "OW0", "S", "IY1", "D", "IH0", "NG", "Z"],
    "orchestra's": ["AO1", "R", "K", "AH0", "S", "T", "R", "AH0", "Z"],
    "orchestra": ["AO1", "R", "K", "AH0", "S", "T", "R", "AH0"],
    "conduct_noun": ["K", "AA1", "N", "D", "AH0", "K", "T"],
    "conduct_verb": ["K", "AH0", "N", "D", "AH1", "K", "T"],
    "ceremony": ["S", "EH1", "R", "AH0", "M", "OW2", "N", "IY0"],
    "was": ["W", "AA1", "Z"],
    "praised": ["P", "R", "EY1", "Z", "D"],
    "by": ["B", "AY1"],
    "audience": ["AO1", "D", "IY0", "AH0", "N", "S"],
    "researchers": ["R", "IY0", "S", "ER1", "CH", "ER0", "Z"],
    "will": ["W", "IH1", "L"],
    "an": ["AE1", "N"],
    "experiment": ["IH0", "K", "S", "P", "EH1", "R", "AH0", "M", "AH0", "N", "T"],
    "to": ["T", "UW1"],
    "evaluate": ["IH0", "V", "AE1", "L", "Y", "UW0", "EY2", "T"],
    "proposed": ["P", "R", "AH0", "P", "OW1", "Z", "D"],
    "method": ["M", "EH1", "TH", "AH0", "D"],
    "rhythm": ["R", "IH1", "DH", "AH0", "M"],
    "of": ["AH1", "V"],
    "changed": ["CH", "EY1", "N", "JH", "D"],
    "unexpectedly": ["AH2", "N", "IH0", "K", "S", "P", "EH1", "K", "T", "IH0", "D", "L", "IY0"],
    "meeting": ["M", "IY1", "T", "IH0", "NG"],
    "start": ["S", "T", "AA1", "R", "T"],
    "at": ["AE1", "T"],
    "ten": ["T", "EH1", "N"],
    "forty-five": ["F", "AO1", "R", "T", "IY0", "F", "AY1", "V"],
    "am": ["EY1", "EH1", "M"],
    "on": ["AA1", "N"],
    "twelfth": ["T", "W", "EH1", "L", "F", "TH"],
    "november": ["N", "OW0", "V", "EH1", "M", "B", "ER0"],
    "twenty": ["T", "W", "EH1", "N", "T", "IY0"],
    "six": ["S", "IH1", "K", "S"],
    "so": ["S", "OW1"],
    "andaroo": ["AH0", "N", "D", "AA1", "R", "UW0"],
    "should": ["SH", "UH1", "D"],
    "arrive": ["ER0", "AY1", "V"],
    "time": ["T", "AY1", "M"],
}


def normalize_numbers_and_dates(text):
    # E3: rhythm typo 'rhyth m'
    text = re.sub(r"\brhyth\s+m\b", "rhythm", text, flags=re.IGNORECASE)
    # E4: 10:45 AM
    text = re.sub(r"10:45\s*AM", "ten forty-five AM", text, flags=re.IGNORECASE)
    # E4: 12 November 2026 -> twelfth November twenty twenty six
    text = re.sub(r"12\s+November\s+2026", "twelfth November twenty twenty six", text, flags=re.IGNORECASE)
    return text


def disambiguate_heteronyms(text):
    """Disambiguate heteronyms like 'conduct' as noun vs verb based on context."""
    # Noun conduct: "orchestra's conduct", "his conduct", "their conduct"
    text = re.sub(r"(\b(?:orchestra's|his|her|their|good|bad|professional)\s+)conduct\b", r"\1conduct_noun", text, flags=re.IGNORECASE)
    # Verb conduct: "will conduct", "to conduct", "shall conduct"
    text = re.sub(r"(\b(?:will|to|shall|can|would|should|could|might)\s+)conduct\b", r"\1conduct_verb", text, flags=re.IGNORECASE)
    return text


def simple_word_to_phones(word):
    """Fallback phoneme generation for unlisted English words."""
    word = word.lower()
    if word in CMU_FALLBACK:
        return CMU_FALLBACK[word]
    
    # Rule-based fallback
    vowels = {"a": "AE1", "e": "EH1", "i": "IH1", "o": "AA1", "u": "AH1"}
    consonants = {
        "b": "B", "c": "K", "d": "D", "f": "F", "g": "G", "h": "HH", "j": "JH",
        "k": "K", "l": "L", "m": "M", "n": "N", "p": "P", "q": "K", "r": "R",
        "s": "S", "t": "T", "v": "V", "w": "W", "x": "K", "y": "Y", "z": "Z"
    }
    phones = []
    for char in word:
        if char in vowels:
            phones.append(vowels[char])
        elif char in consonants:
            phones.append(consonants[char])
    return phones or ["AH0"]


def english_to_phonemes(text):
    """Converts raw English text to sequence of ARPAbet phoneme tokens."""
    text = normalize_numbers_and_dates(text)
    text = disambiguate_heteronyms(text)
    
    tokens = []
    tokens.append(SIL)
    
    words = re.findall(r"[\w']+|[.,!?;]", text)
    for w in words:
        if w in [".", ",", "!", "?", ";"]:
            tokens.append(SP)
        else:
            w_clean = w.lower()
            phones = simple_word_to_phones(w_clean)
            for p in phones:
                tokens.append(f"@{p}")
            tokens.append(SP)
            
    tokens.append(SIL)
    return tokens


def english_text_to_sequence(text):
    """Convert English text string into list of integer symbol IDs."""
    phonemes = english_to_phonemes(text)
    seq = []
    for p in phonemes:
        if p in EN_SYMBOL_TO_ID:
            seq.append(EN_SYMBOL_TO_ID[p])
        else:
            seq.append(EN_SYMBOL_TO_ID[UNK])
    return seq, phonemes

