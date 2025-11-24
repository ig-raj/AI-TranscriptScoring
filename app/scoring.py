# app/scoring.py
"""
Core scoring logic for the Nirmaan transcript tool.

We implement three layers for each high-level rubric criterion:
1. Rule-based scoring (keywords, counts, thresholds from the detailed rubric).
2. NLP semantic similarity:
   - Prefer SentenceTransformer "all-MiniLM-L6-v2" embeddings (if installed).
   - Fall back to bag-of-words cosine similarity if not.
3. Rubric-driven weighting so that category maxima match the "Overall Rubrics" table:

- Content & Structure: 40
- Speech Rate:        10
- Language & Grammar: 20
- Clarity:            15
- Engagement:         15
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List, Tuple
from collections import Counter
import math
import re

# Optional dependencies for better semantic similarity
try:
    import numpy as np
except Exception:  # pragma: no cover - fallback if numpy missing
    np = None  # type: ignore

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:  # pragma: no cover
    SentenceTransformer = None  # type: ignore


# Try to initialise a sentence-transformer model if available
_ST_MODEL = None
if SentenceTransformer is not None and np is not None:  # both needed
    try:
        _ST_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
        print("[scoring] Using SentenceTransformer 'all-MiniLM-L6-v2' for semantic similarity.")
    except Exception as e:  # pragma: no cover
        print(f"[scoring] Could not load SentenceTransformer model: {e}")
        _ST_MODEL = None
else:
    print("[scoring] sentence-transformers or numpy not available; falling back to bag-of-words similarity.")


# -------------------------
# Basic text utilities
# -------------------------

WORD_REGEX = re.compile(r"[a-zA-Z']+")


def simple_tokenize(text: str) -> List[str]:
    """
    Very simple tokenizer: extract alphabetic tokens and lowercase them.
    No external libraries (like nltk) are used.
    """
    return [m.group(0).lower() for m in WORD_REGEX.finditer(text)]


def count_sentences(text: str) -> int:
    """
    Naive sentence count: split on ., ?, ! and count non-empty segments.
    Good enough for our use case.
    """
    parts = re.split(r"[.!?]+", text)
    return sum(1 for p in parts if p.strip())


def cosine_similarity_from_counters(a: Counter, b: Counter) -> float:
    """
    Cosine similarity between two bag-of-words vectors represented as Counters.
    Returns value in [0, 1]. If vectors are zero, returns 0.
    """
    common = set(a.keys()) & set(b.keys())
    dot = sum(a[k] * b[k] for k in common)

    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    denom = norm_a * norm_b
    if denom == 0:
        return 0.0
    return dot / denom


def _semantic_similarity_bow(text: str, description: str) -> float:
    """Old-school bag-of-words similarity, used as fallback."""
    tokens_text = simple_tokenize(text)
    tokens_desc = simple_tokenize(description)
    return cosine_similarity_from_counters(Counter(tokens_text), Counter(tokens_desc))


def semantic_similarity(text: str, description: str) -> float:
    """
    Semantic similarity between transcript and rubric description.

    If sentence-transformers + numpy are installed:
        - use "all-MiniLM-L6-v2" sentence embeddings (handles paraphrases well).
    Otherwise:
        - fall back to bag-of-words cosine similarity.
    Returns a value in [0, 1].
    """
    # Preferred path: sentence-transformers embeddings
    if _ST_MODEL is not None and np is not None:
        try:
            embeddings = _ST_MODEL.encode(
                [text, description],
                normalize_embeddings=True  # L2-normalised; dot product == cosine similarity
            )
            # embeddings is shape (2, dim)
            sim = float(np.dot(embeddings[0], embeddings[1]))
            # Sometimes numerical drift can push slightly outside [-1,1]; clamp + rescale
            sim = max(-1.0, min(1.0, sim))
            # Convert from [-1,1] to [0,1] for consistency with old behaviour
            return (sim + 1.0) / 2.0
        except Exception as e:  # pragma: no cover
            print(f"[scoring] SentenceTransformer similarity failed, falling back to BOW: {e}")
            return _semantic_similarity_bow(text, description)

    # Fallback: bag-of-words cosine
    bow_sim = _semantic_similarity_bow(text, description)
    # Already in [0,1]
    return bow_sim


# -------------------------
# Rubric dataclass
# -------------------------

@dataclass
class CriterionResult:
    name: str
    score: float
    max_score: float
    semantic_similarity: float
    details: Dict[str, Any]
    feedback: str
    # Optional fine-grained breakdown for the UI (e.g. salutation / keywords / flow)
    subcriteria: List[Dict[str, Any]] | None = None


# ----------------------------------------------------
# 1. Content & Structure (max 40)
# ----------------------------------------------------

CONTENT_DESC = (
    "Content & Structure considers whether the introduction has a salutation, "
    "mandatory details (name, age, class, school, family, hobbies/interests), "
    "good-to-have details (origin, ambition, fun fact, strengths), and a good flow "
    "from salutation to closing."
)


def score_content_structure(transcript: str) -> CriterionResult:
    text = transcript.lower()
    tokens = simple_tokenize(text)

    # --- Salutation (0–5) ---
    sal_points = 0
    excellent_phrases = [
        "excited to introduce",
        "feeling great",
        "i am excited",
        "i'm excited",
    ]
    good_salutations = [
        "good morning",
        "good afternoon",
        "good evening",
        "good day",
        "hello everyone",
    ]
    normal_salutations = [
        "hi ",
        "hello ",
        "hi,",
        "hello,",
    ]

    if any(p in text for p in excellent_phrases):
        sal_points = 5
    elif any(p in text for p in good_salutations):
        sal_points = 4
    elif any(p in text for p in normal_salutations):
        sal_points = 2
    else:
        sal_points = 0

    # --- Keyword presence: Must-have (max 20, 5 items * 4) ---
    must_have_groups = {
        "name": ["my name is", "myself ", "i am ", "i'm "],
        "age": ["years old", "year old"],
        "class_school": ["class", "grade", "standard", "school"],
        "family": ["family", "mother", "father", "parents", "brother", "sister"],
        "hobbies": ["hobby", "hobbies", "like to", "love to", "enjoy", "favorite", "favourite"],
    }
    must_have_score = 0
    must_have_found: List[str] = []

    for group_name, patterns in must_have_groups.items():
        if any(p in text for p in patterns):
            must_have_score += 4
            must_have_found.append(group_name)

    # --- Keyword presence: Good-to-have (max 10, 5 items * 2) ---
    good_have_groups = {
        "about_family_extra": ["supportive", "kind", "loving", "caring"],
        "origin_location": ["i am from", "i'm from", "we are from"],
        "ambition_goal": ["i want to become", "my dream is", "my goal is", "i aspire"],
        "fun_fact": ["fun fact", "interesting thing", "one thing people don't know", "something unique"],
        "strengths_achievements": ["achievement", "achievements", "won", "medal", "trophy", "strengths"],
    }
    good_have_score = 0
    good_have_found: List[str] = []

    for group_name, patterns in good_have_groups.items():
        if any(p in text for p in patterns):
            good_have_score += 2
            good_have_found.append(group_name)

    # --- Flow (0–5) ---
    closing_phrases = ["thank you", "thanks for listening", "thank you for listening", "thank you everyone"]
    has_closing = any(p in text for p in closing_phrases)

    # "Basic details" if name and at least one of age/class/school
    has_name = any(p in text for p in must_have_groups["name"])
    has_age_or_class = any(p in text for p in must_have_groups["age"] + must_have_groups["class_school"])

    if sal_points > 0 and has_closing and has_name and has_age_or_class:
        flow_points = 5
        flow_status = "Order followed"
    else:
        flow_points = 0
        flow_status = "Order not clearly followed"

    # Sum all subcomponents according to rubric: 5 + 20 + 10 + 5 = 40
    rule_score = sal_points + must_have_score + good_have_score + flow_points
    max_rule_score = 40.0

    # Semantic similarity with content description (0–1)
    sem_sim = semantic_similarity(transcript, CONTENT_DESC)

    # Combine: mostly rule-based, lightly nudged by semantic similarity
    combined_score = 0.8 * rule_score + 0.2 * (sem_sim * max_rule_score)
    combined_score = max(0.0, min(combined_score, max_rule_score))

    # Feedback
    feedback_parts: List[str] = []
    feedback_parts.append(f"Salutation score: {sal_points}/5.")
    if must_have_score < 20:
        missing = [k for k in must_have_groups.keys() if k not in must_have_found]
        if missing:
            feedback_parts.append(
                "You are missing some mandatory details: " + ", ".join(missing) + "."
            )
    else:
        feedback_parts.append(
            "All mandatory details (name, age, class, school, family, hobbies) are present."
        )

    if good_have_score < 10:
        missing = [k for k in good_have_groups.keys() if k not in good_have_found]
        if missing:
            feedback_parts.append(
                "You could add extra details like: " + ", ".join(missing) + "."
            )
    else:
        feedback_parts.append(
            "Nice extra details about origin, goals, fun facts, or achievements."
        )

    feedback_parts.append(f"Flow: {flow_status}.")
    feedback_parts.append(f"Semantic similarity with the content rubric: {sem_sim:.2f}.")

    feedback = " ".join(feedback_parts)

    details = {
        "salutation_points": sal_points,
        "must_have_score": must_have_score,
        "must_have_found": must_have_found,
        "good_have_score": good_have_score,
        "good_have_found": good_have_found,
        "flow_points": flow_points,
        "flow_status": flow_status,
        "rule_score": rule_score,
        "max_rule_score": max_rule_score,
    }

    # Subcriteria breakdown (for UI like your sketch)
    subcriteria = [
        {"name": "Salutation level", "score": sal_points, "max": 5},
        {"name": "Keyword presence (must + good)", "score": must_have_score + good_have_score, "max": 30},
        {"name": "Flow", "score": flow_points, "max": 5},
    ]

    return CriterionResult(
        name="Content & Structure",
        score=combined_score,
        max_score=max_rule_score,
        semantic_similarity=sem_sim,
        details=details,
        feedback=feedback,
        subcriteria=subcriteria,
    )


# ----------------------------------------------------
# 2. Speech Rate (max 10)
# ----------------------------------------------------

SPEECH_DESC = (
    "Speech rate measures how fast the student speaks based on words per minute. "
    "Ideal range is around 111–140 WPM; too fast or too slow reduces the score."
)


def score_speech_rate(word_count: int, duration_sec: float) -> CriterionResult:
    if duration_sec <= 0:
        duration_sec = 52.0  # safety fallback

    wpm = (word_count / duration_sec) * 60.0

    # Rubric:
    # Too Fast  >161 -> 2
    # Fast      141–160 -> 6
    # Ideal     111–140 -> 10
    # Slow      81–110 -> 6
    # Too slow  <80 -> 2
    if wpm > 161:
        rule_score = 2.0
        band = "Too fast"
    elif 141 <= wpm <= 160:
        rule_score = 6.0
        band = "Fast"
    elif 111 <= wpm <= 140:
        rule_score = 10.0
        band = "Ideal"
    elif 81 <= wpm <= 110:
        rule_score = 6.0
        band = "Slow"
    else:
        rule_score = 2.0
        band = "Too slow"

    max_score = 10.0

    # Semantic similarity (not super meaningful here, but included)
    sem_sim = semantic_similarity(f"{word_count} words", SPEECH_DESC)

    combined_score = 0.9 * rule_score + 0.1 * (sem_sim * max_score)
    combined_score = max(0.0, min(combined_score, max_score))

    feedback = (
        f"Estimated speech rate: {wpm:.1f} WPM, which falls in the '{band}' band "
        f"according to the rubric."
    )

    details = {
        "wpm": wpm,
        "band": band,
        "rule_score": rule_score,
        "max_rule_score": max_score,
    }

    subcriteria = [
        {"name": "Speech as WPM", "score": rule_score, "max": max_score},
    ]

    return CriterionResult(
        name="Speech Rate",
        score=combined_score,
        max_score=max_score,
        semantic_similarity=sem_sim,
        details=details,
        feedback=feedback,
        subcriteria=subcriteria,
    )


# ----------------------------------------------------
# 3. Language & Grammar (max 20)
# ----------------------------------------------------

LANG_DESC = (
    "Language & Grammar evaluates grammar correctness and vocabulary richness. "
    "Grammar is approximated via a heuristic, and vocabulary via Type-Token Ratio (TTR)."
)


def estimate_grammar_score(tokens: List[str]) -> Tuple[float, str]:
    """
    Very rough grammar heuristic:
    - Longer texts with reasonable word lengths and few all-caps or single-letter tokens
      get higher scores.
    - We map the heuristic into 2, 4, 6, 8, or 10 similar to the rubric levels.
    """
    total = len(tokens)
    if total == 0:
        return 2.0, "Very short / empty text; grammar cannot be evaluated."

    short_tokens = sum(1 for t in tokens if len(t) <= 2)
    ratio_short = short_tokens / total

    # Fewer very short tokens -> better grammar-ish impression
    if ratio_short < 0.1:
        score = 10.0
        level = ">0.9"
    elif ratio_short < 0.2:
        score = 8.0
        level = "0.7–0.89"
    elif ratio_short < 0.3:
        score = 6.0
        level = "0.5–0.69"
    elif ratio_short < 0.4:
        score = 4.0
        level = "0.3–0.49"
    else:
        score = 2.0
        level = "<0.3"

    explanation = f"Heuristic grammar level ~{level} based on proportion of very short tokens."
    return score, explanation


def compute_ttr(tokens: List[str]) -> float:
    """Type-Token Ratio = distinct_words / total_words."""
    total = len(tokens)
    if total == 0:
        return 0.0
    distinct = len(set(tokens))
    return distinct / total


def ttr_to_score(ttr: float) -> Tuple[float, str]:
    """
    Map TTR to score according to rubric:
    0.9–1.0  -> 10
    0.7–0.89 -> 8
    0.5–0.69 -> 6
    0.3–0.49 -> 4
    0–0.29   -> 2
    """
    if ttr >= 0.9:
        return 10.0, "0.9–1.0"
    elif ttr >= 0.7:
        return 8.0, "0.7–0.89"
    elif ttr >= 0.5:
        return 6.0, "0.5–0.69"
    elif ttr >= 0.3:
        return 4.0, "0.3–0.49"
    else:
        return 2.0, "0–0.29"


def score_language_grammar(transcript: str) -> CriterionResult:
    tokens = simple_tokenize(transcript)
    max_score = 20.0  # 10 grammar + 10 vocab

    grammar_score, grammar_expl = estimate_grammar_score(tokens)
    ttr_val = compute_ttr(tokens)
    vocab_score, vocab_band = ttr_to_score(ttr_val)

    rule_score = grammar_score + vocab_score

    sem_sim = semantic_similarity(transcript, LANG_DESC)
    combined_score = 0.8 * rule_score + 0.2 * (sem_sim * max_score)
    combined_score = max(0.0, min(combined_score, max_score))

    feedback = (
        f"Estimated grammar score: {grammar_score:.1f}/10. {grammar_expl} "
        f"Vocabulary TTR: {ttr_val:.2f} (band {vocab_band}), score {vocab_score:.1f}/10. "
        f"Semantic similarity with language rubric: {sem_sim:.2f}."
    )

    details = {
        "grammar_score": grammar_score,
        "grammar_explanation": grammar_expl,
        "ttr": ttr_val,
        "vocab_score": vocab_score,
        "vocab_band": vocab_band,
        "rule_score": rule_score,
        "max_rule_score": max_score,
    }

    subcriteria = [
        {"name": "Grammar errors (heuristic)", "score": grammar_score, "max": 10},
        {"name": "Vocabulary richness (TTR)", "score": vocab_score, "max": 10},
    ]

    return CriterionResult(
        name="Language & Grammar",
        score=combined_score,
        max_score=max_score,
        semantic_similarity=sem_sim,
        details=details,
        feedback=feedback,
        subcriteria=subcriteria,
    )


# ----------------------------------------------------
# 4. Clarity (max 15)
# ----------------------------------------------------

CLARITY_DESC = (
    "Clarity looks at filler word usage such as 'um', 'uh', 'like', 'you know'. "
    "Fewer filler words per 100 words give a higher score."
)

FILLER_WORDS = {
    "um", "uh", "like", "you know", "so", "actually", "basically", "right",
    "i mean", "well", "kinda", "sort of", "okay", "ok", "hmm", "ah"
}


def score_clarity(transcript: str) -> CriterionResult:
    text = transcript.lower()
    tokens = simple_tokenize(text)
    total_words = len(tokens) or 1  # avoid div by zero

    # Count filler words with a simple approach: check phrases + single tokens
    filler_count = 0
    for fw in FILLER_WORDS:
        if " " in fw:
            filler_count += text.count(fw)
        else:
            filler_count += sum(1 for t in tokens if t == fw)

    filler_rate = (filler_count / total_words) * 100.0

    # Rubric mapping:
    # 0–3   -> 15
    # 4–6   -> 12
    # 7–9   -> 9
    # 10–12 -> 6
    # 13+   -> 3
    if filler_rate <= 3:
        rule_score = 15.0
        band = "0–3"
    elif filler_rate <= 6:
        rule_score = 12.0
        band = "4–6"
    elif filler_rate <= 9:
        rule_score = 9.0
        band = "7–9"
    elif filler_rate <= 12:
        rule_score = 6.0
        band = "10–12"
    else:
        rule_score = 3.0
        band = "13+"

    max_score = 15.0
    sem_sim = semantic_similarity(transcript, CLARITY_DESC)
    combined_score = 0.85 * rule_score + 0.15 * (sem_sim * max_score)
    combined_score = max(0.0, min(combined_score, max_score))

    feedback = (
        f"Filler word rate: {filler_rate:.2f}% (band {band}). "
        f"Used approx. {filler_count} filler words out of {total_words} total. "
        f"Semantic similarity with clarity rubric: {sem_sim:.2f}."
    )

    details = {
        "filler_count": filler_count,
        "total_words": total_words,
        "filler_rate": filler_rate,
        "band": band,
        "rule_score": rule_score,
        "max_rule_score": max_score,
    }

    subcriteria = [
        {"name": "Filler word rate", "score": rule_score, "max": max_score},
    ]

    return CriterionResult(
        name="Clarity",
        score=combined_score,
        max_score=max_score,
        semantic_similarity=sem_sim,
        details=details,
        feedback=feedback,
        subcriteria=subcriteria,
    )


# ----------------------------------------------------
# 5. Engagement (max 15)
# ----------------------------------------------------

ENGAGEMENT_DESC = (
    "Engagement measures positivity and enthusiasm in the transcript. "
    "We approximate sentiment using a simple lexicon of positive words."
)

POSITIVE_WORDS = {
    "happy", "excited", "glad", "grateful", "love", "enjoy", "interesting",
    "fun", "great", "awesome", "amazing", "proud", "confident", "joy", "joyful"
}


def score_engagement(transcript: str) -> CriterionResult:
    tokens = simple_tokenize(transcript)
    total_words = len(tokens) or 1

    pos_count = sum(1 for t in tokens if t in POSITIVE_WORDS)
    pos_ratio = pos_count / total_words  # 0–1

    # Scale thresholds down (real texts won't be 90% positive words)
    if pos_ratio >= 0.09:
        rule_score = 15.0
        band = ">=0.9 (scaled)"
    elif pos_ratio >= 0.07:
        rule_score = 12.0
        band = "0.7–0.89 (scaled)"
    elif pos_ratio >= 0.05:
        rule_score = 9.0
        band = "0.5–0.69 (scaled)"
    elif pos_ratio >= 0.03:
        rule_score = 6.0
        band = "0.3–0.49 (scaled)"
    else:
        rule_score = 3.0
        band = "<0.3 (scaled)"

    max_score = 15.0
    sem_sim = semantic_similarity(transcript, ENGAGEMENT_DESC)
    combined_score = 0.8 * rule_score + 0.2 * (sem_sim * max_score)
    combined_score = max(0.0, min(combined_score, max_score))

    feedback = (
        f"Positive word ratio: {pos_ratio:.3f} (band {band}), approx. {pos_count} "
        f"positive words out of {total_words}. Semantic similarity with "
        f"engagement rubric: {sem_sim:.2f}."
    )

    details = {
        "positive_count": pos_count,
        "total_words": total_words,
        "positive_ratio": pos_ratio,
        "band": band,
        "rule_score": rule_score,
        "max_rule_score": max_score,
    }

    subcriteria = [
        {"name": "Sentiment positivity", "score": rule_score, "max": max_score},
    ]

    return CriterionResult(
        name="Engagement",
        score=combined_score,
        max_score=max_score,
        semantic_similarity=sem_sim,
        details=details,
        feedback=feedback,
        subcriteria=subcriteria,
    )


# ----------------------------------------------------
# Public API
# ----------------------------------------------------

def score_transcript(transcript: str, duration_sec: float = 52.0) -> Dict[str, Any]:
    """
    High-level scoring entrypoint.

    - Computes:
        * word count
        * sentence count
        * WPM (using provided duration, default 52s)
    - Scores each high-level rubric criterion (Content & Structure, Speech Rate,
      Language & Grammar, Clarity, Engagement).
    - Returns:
        {
          "overall_score": float,
          "words": int,
          "sentences": int,
          "wpm": float,
          "criteria": [
             {
               "name": ...,
               "score": ...,
               "max": ...,
               "semantic_similarity": ...,
               "details": {...},
               "feedback": "...",
               "subcriteria": [{ name, score, max }, ...]
             },
             ...
          ]
        }
    """
    transcript = transcript.strip()
    tokens = simple_tokenize(transcript)
    word_count = len(tokens)
    sentence_count = count_sentences(transcript)

    # Category scores
    content_res = score_content_structure(transcript)
    speech_res = score_speech_rate(word_count, duration_sec)
    lang_res = score_language_grammar(transcript)
    clarity_res = score_clarity(transcript)
    engage_res = score_engagement(transcript)

    criteria_results = [content_res, speech_res, lang_res, clarity_res, engage_res]

    overall_score = sum(c.score for c in criteria_results)
    max_overall = sum(c.max_score for c in criteria_results)
    # Normalize to 0–100 (should already be ~0–100, but normalize safely)
    if max_overall > 0:
        overall_score = (overall_score / max_overall) * 100.0
    else:
        overall_score = 0.0

    # Compute WPM again for output
    if duration_sec <= 0:
        duration_sec = 52.0
    wpm = (word_count / duration_sec) * 60.0

    return {
        "overall_score": round(overall_score, 1),
        "words": word_count,
        "sentences": sentence_count,
        "wpm": round(wpm, 1),
        "criteria": [
            {
                "name": c.name,
                "score": round(c.score, 2),
                "max": c.max_score,
                "semantic_similarity": round(c.semantic_similarity, 3),
                "details": c.details,
                "feedback": c.feedback,
                "subcriteria": c.subcriteria,
            }
            for c in criteria_results
        ],
    }
