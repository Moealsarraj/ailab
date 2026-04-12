"""Deterministic Arabic NLP metrics — no AI calls, instant results.

Provides BLEU, exact match, character overlap, length ratio, and
Arabic-specific metrics (tashkeel preservation, Arabic char ratio).
"""
import re
import math
from collections import Counter


def compute_all_metrics(ai_response: str, reference: str) -> dict:
    """Compute all deterministic metrics between AI response and reference."""
    ai = ai_response.strip()
    ref = reference.strip()

    return {
        "bleu_score": bleu(ai, ref),
        "exact_match": exact_match(ai, ref),
        "char_overlap": char_overlap(ai, ref),
        "word_overlap": word_overlap(ai, ref),
        "length_ratio": length_ratio(ai, ref),
        "arabic_char_ratio": arabic_char_ratio(ai),
        "tashkeel_count": tashkeel_count(ai),
        "ref_tashkeel_count": tashkeel_count(ref),
    }


# ── BLEU Score (sentence-level, smoothed) ──

def _ngrams(tokens: list, n: int) -> Counter:
    return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))


def bleu(candidate: str, reference: str, max_n: int = 4) -> float:
    """Compute smoothed sentence-level BLEU score (0-100)."""
    cand_tokens = candidate.split()
    ref_tokens = reference.split()

    if not cand_tokens or not ref_tokens:
        return 0.0

    # Brevity penalty
    bp = min(1.0, math.exp(1 - len(ref_tokens) / max(len(cand_tokens), 1)))

    # Modified precision for each n-gram order
    log_avg = 0.0
    weights = 1.0 / min(max_n, len(cand_tokens))  # uniform weights

    for n in range(1, min(max_n, len(cand_tokens)) + 1):
        cand_ngrams = _ngrams(cand_tokens, n)
        ref_ngrams = _ngrams(ref_tokens, n)

        # Clipped counts
        clipped = 0
        total = 0
        for ng, count in cand_ngrams.items():
            clipped += min(count, ref_ngrams.get(ng, 0))
            total += count

        # Add-1 smoothing
        precision = (clipped + 1) / (total + 1)
        log_avg += weights * math.log(precision)

    return round(bp * math.exp(log_avg) * 100, 1)


# ── Exact Match ──

def exact_match(candidate: str, reference: str) -> bool:
    """Check if candidate exactly matches reference (normalized whitespace)."""
    return _normalize(candidate) == _normalize(reference)


def _normalize(text: str) -> str:
    """Normalize Arabic text for comparison."""
    # Remove tashkeel (diacritics)
    text = re.sub(r'[\u064B-\u065F\u0670]', '', text)
    # Normalize alef variants
    text = re.sub(r'[إأآا]', 'ا', text)
    # Normalize taa marbuta
    text = text.replace('ة', 'ه')
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ── Character Overlap (Jaccard) ──

def char_overlap(candidate: str, reference: str) -> float:
    """Character-level Jaccard similarity (0-100)."""
    c_set = set(candidate.replace(' ', ''))
    r_set = set(reference.replace(' ', ''))
    if not c_set and not r_set:
        return 100.0
    if not c_set or not r_set:
        return 0.0
    intersection = c_set & r_set
    union = c_set | r_set
    return round(len(intersection) / len(union) * 100, 1)


# ── Word Overlap (F1) ──

def word_overlap(candidate: str, reference: str) -> float:
    """Word-level F1 score (0-100)."""
    c_words = set(_normalize(candidate).split())
    r_words = set(_normalize(reference).split())
    if not c_words and not r_words:
        return 100.0
    if not c_words or not r_words:
        return 0.0
    common = c_words & r_words
    precision = len(common) / len(c_words) if c_words else 0
    recall = len(common) / len(r_words) if r_words else 0
    if precision + recall == 0:
        return 0.0
    f1 = 2 * precision * recall / (precision + recall)
    return round(f1 * 100, 1)


# ── Length Ratio ──

def length_ratio(candidate: str, reference: str) -> float:
    """Ratio of candidate length to reference length (1.0 = same length)."""
    c_len = len(candidate.split())
    r_len = len(reference.split())
    if r_len == 0:
        return 0.0
    return round(c_len / r_len, 2)


# ── Arabic-specific metrics ──

_ARABIC_RANGE = re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]')
_TASHKEEL = re.compile(r'[\u064B-\u065F\u0670]')


def arabic_char_ratio(text: str) -> float:
    """Percentage of characters that are Arabic script (0-100)."""
    chars = text.replace(' ', '')
    if not chars:
        return 0.0
    arabic_count = len(_ARABIC_RANGE.findall(chars))
    return round(arabic_count / len(chars) * 100, 1)


def tashkeel_count(text: str) -> int:
    """Count of tashkeel (diacritical) marks in text."""
    return len(_TASHKEEL.findall(text))
