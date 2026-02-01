"""
Advanced Differentiation Scoring for CEI Quality Pipeline.

This module provides enhanced scoring mechanisms for better differentiation
between annotation quality levels, including:
- Plutchik wheel emotion distance calculation
- Per-scenario weighted agreement scoring
- Dwell time integration into quality scores
- Entropy-based disagreement quantification
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from math import log2
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from cei_quality.config import CEIConfig
    from cei_quality.models import AnnotationData


# =============================================================================
# Plutchik Wheel Emotion Distance
# =============================================================================

# Plutchik's 8 primary emotions arranged on the wheel (clockwise)
# Adjacent emotions are distance 1, opposite emotions are distance 4
PLUTCHIK_WHEEL_ORDER = [
    "joy",
    "trust",
    "fear",
    "surprise",
    "sadness",
    "disgust",
    "anger",
    "anticipation",
]

# Pre-computed distance matrix (0-4 scale)
EMOTION_DISTANCE: Dict[str, Dict[str, int]] = {}

def _init_emotion_distance() -> None:
    """Initialize the Plutchik wheel distance matrix."""
    n = len(PLUTCHIK_WHEEL_ORDER)
    for i, e1 in enumerate(PLUTCHIK_WHEEL_ORDER):
        EMOTION_DISTANCE[e1] = {}
        for j, e2 in enumerate(PLUTCHIK_WHEEL_ORDER):
            # Minimum distance on circular wheel
            dist = min(abs(i - j), n - abs(i - j))
            EMOTION_DISTANCE[e1][e2] = dist

_init_emotion_distance()


def get_emotion_distance(emotion1: str, emotion2: str) -> int:
    """
    Get Plutchik wheel distance between two emotions.

    Args:
        emotion1: First emotion (case-insensitive)
        emotion2: Second emotion (case-insensitive)

    Returns:
        Distance 0-4 where:
        - 0: Same emotion
        - 1: Adjacent (e.g., joy↔trust)
        - 2: Two steps apart (e.g., joy↔fear)
        - 3: Three steps apart (e.g., joy↔surprise)
        - 4: Opposite (e.g., joy↔sadness)

    Examples:
        >>> get_emotion_distance("joy", "joy")
        0
        >>> get_emotion_distance("joy", "trust")
        1
        >>> get_emotion_distance("joy", "sadness")
        4
    """
    e1 = emotion1.lower().strip()
    e2 = emotion2.lower().strip()

    if e1 not in EMOTION_DISTANCE or e2 not in EMOTION_DISTANCE:
        return 4  # Maximum distance for unknown emotions

    return EMOTION_DISTANCE[e1][e2]


def compute_emotion_dispersion(emotions: List[str]) -> float:
    """
    Compute dispersion score for a set of emotions.

    Higher dispersion = more disagreement on fundamentally different emotions.

    Args:
        emotions: List of emotion labels from multiple annotators

    Returns:
        Dispersion score 0.0-1.0 where:
        - 0.0: All same emotion (unanimous)
        - ~0.25: Adjacent emotions (minor disagreement)
        - ~0.5: Moderate distance
        - 1.0: Maximum distance (opposite emotions)

    Examples:
        >>> compute_emotion_dispersion(["joy", "joy", "joy"])
        0.0
        >>> compute_emotion_dispersion(["joy", "trust", "anticipation"])
        0.25  # All adjacent to joy
        >>> compute_emotion_dispersion(["joy", "sadness", "anger"])
        0.75  # Mix of opposite emotions
    """
    if len(emotions) < 2:
        return 0.0

    # Compute average pairwise distance
    total_distance = 0
    pair_count = 0

    for i in range(len(emotions)):
        for j in range(i + 1, len(emotions)):
            total_distance += get_emotion_distance(emotions[i], emotions[j])
            pair_count += 1

    if pair_count == 0:
        return 0.0

    avg_distance = total_distance / pair_count

    # Normalize to 0-1 (max distance is 4)
    return avg_distance / 4.0


# =============================================================================
# Per-Scenario Agreement Metrics
# =============================================================================

@dataclass
class ScenarioAgreementMetrics:
    """
    Detailed agreement metrics for a single scenario.

    Provides multiple dimensions of agreement analysis beyond
    simple majority voting.
    """

    scenario_id: int
    subtype: str

    # Basic agreement
    emotions: List[str] = field(default_factory=list)
    has_majority: bool = False
    majority_emotion: Optional[str] = None
    majority_count: int = 0

    # Plutchik wheel metrics
    emotion_dispersion: float = 0.0  # 0-1, higher = more scattered
    max_pairwise_distance: int = 0  # 0-4
    contains_opposites: bool = False  # True if any pair has distance 4

    # Information-theoretic metrics
    emotion_entropy: float = 0.0  # Bits of uncertainty
    normalized_entropy: float = 0.0  # 0-1

    # VAD agreement
    vad_variance: float = 0.0  # Avg variance across V, A, D
    vad_range: float = 0.0  # Avg range across V, A, D

    # Timing
    mean_lead_time: float = 0.0
    min_lead_time: float = 0.0
    lead_time_variance: float = 0.0
    has_rushing_annotator: bool = False

    # Composite score (0-1, higher = better agreement)
    agreement_score: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "scenario_id": self.scenario_id,
            "subtype": self.subtype,
            "emotions": self.emotions,
            "has_majority": self.has_majority,
            "majority_emotion": self.majority_emotion,
            "majority_count": self.majority_count,
            "emotion_dispersion": round(self.emotion_dispersion, 4),
            "max_pairwise_distance": self.max_pairwise_distance,
            "contains_opposites": self.contains_opposites,
            "emotion_entropy": round(self.emotion_entropy, 4),
            "normalized_entropy": round(self.normalized_entropy, 4),
            "vad_variance": round(self.vad_variance, 4),
            "vad_range": round(self.vad_range, 4),
            "mean_lead_time": round(self.mean_lead_time, 2),
            "min_lead_time": round(self.min_lead_time, 2),
            "has_rushing_annotator": self.has_rushing_annotator,
            "agreement_score": round(self.agreement_score, 4),
        }


def compute_emotion_entropy(emotions: List[str]) -> tuple[float, float]:
    """
    Compute Shannon entropy of emotion distribution.

    Args:
        emotions: List of emotion labels

    Returns:
        Tuple of (raw_entropy, normalized_entropy)
        - raw_entropy: Bits of uncertainty (0 to log2(n))
        - normalized_entropy: 0-1 scale
    """
    if not emotions:
        return 0.0, 0.0

    counts = Counter(emotions)
    n = len(emotions)

    entropy = 0.0
    for count in counts.values():
        if count > 0:
            p = count / n
            entropy -= p * log2(p)

    # Maximum possible entropy is log2(n) if all different
    max_entropy = log2(min(n, 8))  # Cap at 8 (Plutchik emotions)
    normalized = entropy / max_entropy if max_entropy > 0 else 0.0

    return entropy, normalized


def compute_scenario_agreement(
    scenario_id: int,
    subtype: str,
    annotations: List["AnnotationData"],
    config: "CEIConfig",
) -> ScenarioAgreementMetrics:
    """
    Compute comprehensive agreement metrics for a scenario.

    Args:
        scenario_id: Scenario identifier
        subtype: Pragmatic subtype name
        annotations: List of annotations for this scenario
        config: CEI configuration

    Returns:
        ScenarioAgreementMetrics with all computed values
    """
    metrics = ScenarioAgreementMetrics(
        scenario_id=scenario_id,
        subtype=subtype,
    )

    if not annotations:
        return metrics

    # Extract emotions
    emotions = [a.plutchik_emotion.lower().strip() for a in annotations if a.plutchik_emotion]
    metrics.emotions = emotions

    if not emotions:
        return metrics

    # Basic majority
    counts = Counter(emotions)
    most_common = counts.most_common(1)[0]
    metrics.majority_count = most_common[1]

    if most_common[1] >= 2:  # 2/3 majority
        metrics.has_majority = True
        metrics.majority_emotion = most_common[0]

    # Plutchik wheel metrics
    metrics.emotion_dispersion = compute_emotion_dispersion(emotions)

    max_dist = 0
    for i in range(len(emotions)):
        for j in range(i + 1, len(emotions)):
            dist = get_emotion_distance(emotions[i], emotions[j])
            max_dist = max(max_dist, dist)
    metrics.max_pairwise_distance = max_dist
    metrics.contains_opposites = max_dist >= 4

    # Entropy
    entropy, norm_entropy = compute_emotion_entropy(emotions)
    metrics.emotion_entropy = entropy
    metrics.normalized_entropy = norm_entropy

    # VAD variance
    v_vals = []
    a_vals = []
    d_vals = []

    for ann in annotations:
        v_vals.append(config.get_vad_numeric("valence", ann.valence))
        a_vals.append(config.get_vad_numeric("arousal", ann.arousal))
        d_vals.append(config.get_vad_numeric("dominance", ann.dominance))

    if v_vals:
        import numpy as np
        variances = [np.var(v_vals), np.var(a_vals), np.var(d_vals)]
        ranges = [
            max(v_vals) - min(v_vals),
            max(a_vals) - min(a_vals),
            max(d_vals) - min(d_vals),
        ]
        metrics.vad_variance = float(np.mean(variances))
        metrics.vad_range = float(np.mean(ranges))

    # Timing
    lead_times = [a.lead_time for a in annotations if a.lead_time is not None]
    if lead_times:
        import numpy as np
        metrics.mean_lead_time = float(np.mean(lead_times))
        metrics.min_lead_time = float(min(lead_times))
        metrics.lead_time_variance = float(np.var(lead_times))

        # Check for rushing (configurable threshold)
        min_threshold = config.quality.lead_time.suspiciously_fast
        metrics.has_rushing_annotator = metrics.min_lead_time < min_threshold

    # Composite agreement score
    # Formula: base_score - dispersion_penalty - entropy_penalty - rushing_penalty
    base = 1.0

    # Dispersion penalty (0-0.3)
    dispersion_penalty = metrics.emotion_dispersion * 0.3

    # Entropy penalty (0-0.2)
    entropy_penalty = metrics.normalized_entropy * 0.2

    # Opposite emotions penalty (flat 0.2 if contains opposites)
    opposite_penalty = 0.2 if metrics.contains_opposites else 0.0

    # Rushing penalty (0.1 if any rushing annotator)
    rushing_penalty = 0.1 if metrics.has_rushing_annotator else 0.0

    # No majority penalty (0.2)
    no_majority_penalty = 0.2 if not metrics.has_majority else 0.0

    metrics.agreement_score = max(0.0, base - dispersion_penalty - entropy_penalty
                                   - opposite_penalty - rushing_penalty - no_majority_penalty)

    return metrics


# =============================================================================
# Differentiation Score Calculator
# =============================================================================

@dataclass
class DifferentiationScore:
    """
    Enhanced quality score with better differentiation.

    Extends the base quality score with additional metrics
    that provide finer-grained quality assessment.
    """

    scenario_id: int
    subtype: str

    # Base quality score (from QualityScoreAggregator)
    base_quality_score: float = 1.0

    # Agreement metrics
    agreement_metrics: Optional[ScenarioAgreementMetrics] = None

    # Differentiation factors
    dispersion_factor: float = 1.0  # Multiplier based on emotion dispersion
    timing_factor: float = 1.0  # Multiplier based on timing quality
    consistency_factor: float = 1.0  # Multiplier based on VAD consistency

    # Final differentiated score
    differentiated_score: float = 1.0

    # Review priority (0-100, higher = more urgent)
    priority_rank: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "scenario_id": self.scenario_id,
            "subtype": self.subtype,
            "base_quality_score": round(self.base_quality_score, 4),
            "agreement_metrics": self.agreement_metrics.to_dict() if self.agreement_metrics else None,
            "dispersion_factor": round(self.dispersion_factor, 4),
            "timing_factor": round(self.timing_factor, 4),
            "consistency_factor": round(self.consistency_factor, 4),
            "differentiated_score": round(self.differentiated_score, 4),
            "priority_rank": self.priority_rank,
        }


class DifferentiationCalculator:
    """
    Calculator for enhanced differentiation scores.

    Combines multiple factors to produce scores with better
    separation between quality levels.
    """

    def __init__(self, config: "CEIConfig") -> None:
        self.config = config

        # Weights for combining factors
        self.dispersion_weight = 0.25
        self.timing_weight = 0.15
        self.consistency_weight = 0.15
        self.base_weight = 0.45

    def compute_differentiated_score(
        self,
        scenario_id: int,
        subtype: str,
        base_quality_score: float,
        annotations: List["AnnotationData"],
    ) -> DifferentiationScore:
        """
        Compute differentiated score for a scenario.

        Args:
            scenario_id: Scenario identifier
            subtype: Pragmatic subtype name
            base_quality_score: Score from base quality aggregator
            annotations: Annotations for this scenario

        Returns:
            DifferentiationScore with all factors computed
        """
        score = DifferentiationScore(
            scenario_id=scenario_id,
            subtype=subtype,
            base_quality_score=base_quality_score,
        )

        # Compute agreement metrics
        metrics = compute_scenario_agreement(
            scenario_id, subtype, annotations, self.config
        )
        score.agreement_metrics = metrics

        # Dispersion factor: penalize scattered emotions
        # 1.0 = no dispersion, 0.5 = maximum dispersion
        score.dispersion_factor = 1.0 - (metrics.emotion_dispersion * 0.5)

        # Timing factor: penalize rushing
        if metrics.has_rushing_annotator:
            # More severe penalty if multiple annotators rushed
            rushing_count = sum(
                1 for a in annotations
                if a.lead_time and a.lead_time < self.config.quality.lead_time.suspiciously_fast
            )
            score.timing_factor = max(0.5, 1.0 - (rushing_count * 0.15))

        # Consistency factor: penalize high VAD variance
        # Normalize variance (assume max reasonable variance is 2.0)
        normalized_var = min(metrics.vad_variance / 2.0, 1.0)
        score.consistency_factor = 1.0 - (normalized_var * 0.3)

        # Compute differentiated score
        score.differentiated_score = (
            self.base_weight * base_quality_score +
            self.dispersion_weight * score.dispersion_factor * base_quality_score +
            self.timing_weight * score.timing_factor * base_quality_score +
            self.consistency_weight * score.consistency_factor * base_quality_score
        )

        # Clamp to 0-1
        score.differentiated_score = max(0.0, min(1.0, score.differentiated_score))

        return score

    def rank_scenarios(
        self,
        scores: List[DifferentiationScore],
    ) -> List[DifferentiationScore]:
        """
        Rank scenarios by priority (lowest score = highest priority).

        Args:
            scores: List of differentiation scores

        Returns:
            Sorted list with priority_rank set (1 = highest priority)
        """
        # Sort by differentiated score ascending (worst first)
        sorted_scores = sorted(scores, key=lambda s: s.differentiated_score)

        # Assign ranks
        for i, score in enumerate(sorted_scores):
            score.priority_rank = i + 1

        return sorted_scores
