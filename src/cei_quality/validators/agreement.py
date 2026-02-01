"""
Stage 1C: Inter-Annotator Agreement Validation

Validates consistency across multiple annotators for the same scenarios.

Checks performed:
- All 3 annotators have annotated each scenario
- Emotion agreement (identifies complete disagreements)
- VAD rating agreement (identifies high variance)
- Computes Fleiss' kappa for overall agreement
- Computes weighted Cohen's kappa for ordinal VAD scales
- Generates confusion matrix for emotion disagreement patterns

Key insight: Some disagreement is expected and valuable for studying
pragmatic ambiguity. We flag only extreme disagreements.
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

from cei_quality.models import (
    AnnotatorFile,
    AnnotationData,
    PragmaticSubtype,
    QualityFlag,
    QualityIssue,
    ScenarioData,
)

if TYPE_CHECKING:
    from cei_quality.config import CEIConfig

logger = logging.getLogger(__name__)


class InterAnnotatorValidator:
    """
    Stage 1C: Validate agreement across annotators.

    This validator compares annotations from different annotators
    for the same scenarios to identify disagreements and compute
    agreement metrics.

    Note: Moderate disagreement is expected for pragmatic ambiguity
    tasks. We only flag cases of extreme disagreement.

    Example:
        >>> validator = InterAnnotatorValidator(config)
        >>> issues = validator.validate_subtype(
        ...     subtype=PragmaticSubtype.SARCASM_IRONY,
        ...     annotator_files={...},
        ...     scenarios={...},
        ...     annotations={...}
        ... )
        >>> kappa = validator.compute_fleiss_kappa(annotations)
    """

    def __init__(self, config: CEIConfig) -> None:
        """
        Initialize agreement validator.

        Args:
            config: CEI configuration instance
        """
        self.config = config
        self._quality = config.quality

    def validate_subtype(
        self,
        subtype: PragmaticSubtype,
        annotator_files: Dict[str, AnnotatorFile],
        scenarios: Dict[int, ScenarioData],
        annotations: Dict[int, List[AnnotationData]],
    ) -> Dict[int, List[QualityIssue]]:
        """
        Validate inter-annotator agreement for a subtype.

        Args:
            subtype: The pragmatic subtype being validated
            annotator_files: Dict mapping annotator_name -> AnnotatorFile
            scenarios: Dict mapping scenario_id -> ScenarioData
            annotations: Dict mapping scenario_id -> List[AnnotationData]

        Returns:
            Dict mapping scenario_id -> list of agreement issues
        """
        issues_by_scenario: Dict[int, List[QualityIssue]] = defaultdict(list)

        expected_annotators = set(annotator_files.keys())
        n_annotators = len(expected_annotators)

        if n_annotators < 2:
            logger.warning(f"Subtype {subtype.value} has only {n_annotators} annotators")
            return dict(issues_by_scenario)

        # Check each scenario that exists in the annotations
        # NOTE: We iterate over actual scenario IDs, not assumed 1-N range.
        # This handles non-contiguous scenario IDs correctly.
        all_scenario_ids = set(annotations.keys()) | set(scenarios.keys())

        for scenario_id in sorted(all_scenario_ids):
            scenario_annotations = annotations.get(scenario_id, [])

            # Check for missing cross-annotations
            present_annotators = {a.annotator_name for a in scenario_annotations}
            missing_annotators = expected_annotators - present_annotators

            if missing_annotators:
                issues_by_scenario[scenario_id].append(
                    QualityIssue(
                        flag=QualityFlag.MISSING_CROSS_ANNOTATION,
                        severity="critical",
                        message=f"Scenario {scenario_id} missing annotations from: {missing_annotators}",
                        details={
                            "scenario_id": scenario_id,
                            "subtype": subtype.value,
                            "expected_annotators": list(expected_annotators),
                            "present_annotators": list(present_annotators),
                            "missing_annotators": list(missing_annotators),
                        },
                    )
                )

            # Check agreement if we have at least 2 annotations
            if len(scenario_annotations) >= 2:
                # Check emotion agreement
                emotion_issues = self._check_emotion_agreement(
                    scenario_id, scenario_annotations, subtype
                )
                issues_by_scenario[scenario_id].extend(emotion_issues)

                # Check VAD agreement
                vad_issues = self._check_vad_agreement(scenario_id, scenario_annotations)
                issues_by_scenario[scenario_id].extend(vad_issues)

        return dict(issues_by_scenario)

    def _check_emotion_agreement(
        self,
        scenario_id: int,
        annotations: List[AnnotationData],
        subtype: PragmaticSubtype,
    ) -> List[QualityIssue]:
        """Check agreement on Plutchik emotion."""
        issues: List[QualityIssue] = []

        # Get emotions from all annotators
        emotions = [a.plutchik_emotion.lower().strip() for a in annotations if a.plutchik_emotion]

        if len(emotions) < 2:
            return issues

        unique_emotions = set(emotions)
        counts = Counter(emotions)
        n_annotators = len(emotions)

        # Complete disagreement: all different emotions
        if len(unique_emotions) == n_annotators and n_annotators >= 3:
            issues.append(
                QualityIssue(
                    flag=QualityFlag.NO_MAJORITY_EMOTION,
                    severity="major",
                    message=f"Complete emotion disagreement: {unique_emotions}",
                    details={
                        "scenario_id": scenario_id,
                        "subtype": subtype.value,
                        "emotions": emotions,
                        "unique_emotions": list(unique_emotions),
                        "by_annotator": {a.annotator_name: a.plutchik_emotion for a in annotations},
                    },
                )
            )

        # No majority: no emotion has 2+ votes
        elif len(unique_emotions) > 1:
            majority_count = counts.most_common(1)[0][1]
            agreement_ratio = majority_count / n_annotators

            # Only flag if no majority (< 2/3 agreement with 3 annotators)
            if majority_count < 2 and n_annotators >= 3:
                issues.append(
                    QualityIssue(
                        flag=QualityFlag.HIGH_EMOTION_DISAGREEMENT,
                        severity="minor",
                        message=f"Low emotion agreement ({agreement_ratio:.0%}): {unique_emotions}",
                        details={
                            "scenario_id": scenario_id,
                            "emotions": emotions,
                            "agreement_ratio": agreement_ratio,
                            "majority_emotion": counts.most_common(1)[0][0],
                            "by_annotator": {
                                a.annotator_name: a.plutchik_emotion for a in annotations
                            },
                        },
                    )
                )

        return issues

    def _check_vad_agreement(
        self,
        scenario_id: int,
        annotations: List[AnnotationData],
    ) -> List[QualityIssue]:
        """Check agreement on VAD ratings."""
        issues: List[QualityIssue] = []

        # Get numeric VAD values
        v_vals: List[float] = []
        a_vals: List[float] = []
        d_vals: List[float] = []

        for ann in annotations:
            vad = ann.get_vad_numeric(self.config)
            v_vals.append(vad["v"])
            a_vals.append(vad["a"])
            d_vals.append(vad["d"])

        if len(v_vals) < 2:
            return issues

        # Get thresholds from config
        vad_config = self._quality.vad_disagreement
        high_threshold = vad_config.high
        moderate_threshold = vad_config.moderate

        # Check each dimension
        dimensions = [
            ("valence", v_vals, "sl_v"),
            ("arousal", a_vals, "sl_a"),
            ("dominance", d_vals, "sl_d"),
        ]

        for dim_name, vals, field_name in dimensions:
            val_range = max(vals) - min(vals)

            if val_range > high_threshold:
                issues.append(
                    QualityIssue(
                        flag=QualityFlag.HIGH_VAD_DISAGREEMENT,
                        severity="major",
                        message=f"High {dim_name} disagreement (range={val_range:.2f})",
                        details={
                            "scenario_id": scenario_id,
                            "dimension": dim_name,
                            "field": field_name,
                            "values": vals,
                            "range": val_range,
                            "threshold": high_threshold,
                            "by_annotator": {
                                a.annotator_name: getattr(a, dim_name) for a in annotations
                            },
                        },
                    )
                )
            elif val_range > moderate_threshold:
                issues.append(
                    QualityIssue(
                        flag=QualityFlag.HIGH_VAD_DISAGREEMENT,
                        severity="minor",
                        message=f"Moderate {dim_name} disagreement (range={val_range:.2f})",
                        details={
                            "scenario_id": scenario_id,
                            "dimension": dim_name,
                            "values": vals,
                            "range": val_range,
                            "threshold": moderate_threshold,
                        },
                    )
                )

        return issues

    def compute_fleiss_kappa(
        self,
        annotations_by_scenario: Dict[int, List[AnnotationData]],
    ) -> float:
        """
        Compute Fleiss' kappa for emotion agreement.

        Fleiss' kappa measures agreement among multiple raters
        assigning categorical ratings to items.

        Args:
            annotations_by_scenario: Dict mapping scenario_id -> annotations

        Returns:
            Fleiss' kappa coefficient (-1 to 1, where 1 = perfect agreement)
        """
        valid_emotions = self.config.valid_labels.plutchik_emotions
        n_categories = len(valid_emotions)
        emotion_to_idx = {e.lower(): i for i, e in enumerate(sorted(valid_emotions))}

        # Build ratings matrix: (n_items x n_categories)
        # Each cell = count of raters assigning that category to that item
        ratings_matrix: List[List[int]] = []

        for scenario_id in sorted(annotations_by_scenario.keys()):
            annotations = annotations_by_scenario[scenario_id]

            if not annotations:
                continue

            row = [0] * n_categories

            for ann in annotations:
                emotion = ann.plutchik_emotion.lower().strip()
                if emotion in emotion_to_idx:
                    row[emotion_to_idx[emotion]] += 1

            # Only include if we have at least one valid rating
            if sum(row) > 0:
                ratings_matrix.append(row)

        if not ratings_matrix:
            return 0.0

        matrix = np.array(ratings_matrix, dtype=float)
        n_items = matrix.shape[0]
        n_raters = matrix.sum(axis=1)[0]  # Assume same number of raters per item

        if n_raters < 2:
            return 0.0

        # Proportion assigned to each category (p_j)
        p_j = matrix.sum(axis=0) / (n_items * n_raters)

        # Extent of agreement for each item (P_i)
        # P_i = (sum of n_ij^2 - n_raters) / (n_raters * (n_raters - 1))
        P_i = (matrix**2).sum(axis=1) - n_raters
        P_i = P_i / (n_raters * (n_raters - 1))

        # Mean observed agreement (P_bar)
        P_bar = float(P_i.mean())

        # Expected agreement by chance (P_e_bar)
        P_e_bar = float((p_j**2).sum())

        # Fleiss' kappa
        if P_e_bar >= 1.0:
            return 1.0  # Edge case: perfect agreement

        kappa = (P_bar - P_e_bar) / (1 - P_e_bar)

        return float(kappa)

    def compute_pairwise_agreement(
        self,
        annotations_by_scenario: Dict[int, List[AnnotationData]],
    ) -> Dict[Tuple[str, str], float]:
        """
        Compute pairwise agreement between annotators.

        Useful for identifying outlier annotators who systematically
        disagree with others.

        Args:
            annotations_by_scenario: Dict mapping scenario_id -> annotations

        Returns:
            Dict mapping (annotator1, annotator2) -> agreement rate
        """
        # Collect agreements per annotator pair
        pair_agreements: Dict[Tuple[str, str], List[bool]] = defaultdict(list)

        for scenario_id, annotations in annotations_by_scenario.items():
            if len(annotations) < 2:
                continue

            # Check each pair
            for i, ann1 in enumerate(annotations):
                for ann2 in annotations[i + 1 :]:
                    pair = tuple(sorted([ann1.annotator_name, ann2.annotator_name]))
                    agrees = ann1.plutchik_emotion.lower() == ann2.plutchik_emotion.lower()
                    pair_agreements[pair].append(agrees)

        # Compute agreement rates
        agreement_rates: Dict[Tuple[str, str], float] = {}

        for pair, agreements in pair_agreements.items():
            if agreements:
                agreement_rates[pair] = sum(agreements) / len(agreements)

        return agreement_rates

    def detect_systematic_outliers(
        self,
        annotations_by_scenario: Dict[int, List[AnnotationData]],
        threshold: float = 0.3,
    ) -> Dict[str, Any]:
        """
        Detect annotators who systematically disagree with the other annotators.

        An annotator is flagged as a systematic outlier if:
        1. Their average pairwise agreement with others is significantly lower
        2. They form the minority opinion in a high percentage of 2-1 splits

        Args:
            annotations_by_scenario: Dict mapping scenario_id -> annotations
            threshold: Agreement threshold below which to flag (default 0.3)

        Returns:
            Dict containing:
            - outlier_annotators: List of annotators with low agreement
            - annotator_stats: Per-annotator statistics
            - minority_patterns: Scenarios where annotator was the lone dissenter
        """
        # Collect per-annotator statistics
        annotator_agreement_rates: Dict[str, List[float]] = defaultdict(list)
        annotator_minority_count: Dict[str, int] = defaultdict(int)
        annotator_total_scenarios: Dict[str, int] = defaultdict(int)
        minority_patterns: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        pairwise_agreements = self.compute_pairwise_agreement(annotations_by_scenario)

        # Build per-annotator agreement rates from pairwise
        for (ann1, ann2), rate in pairwise_agreements.items():
            annotator_agreement_rates[ann1].append(rate)
            annotator_agreement_rates[ann2].append(rate)

        # Check for 2-1 splits (one annotator is the minority)
        for scenario_id, annotations in annotations_by_scenario.items():
            if len(annotations) != 3:
                continue

            emotions = [(a.annotator_name, a.plutchik_emotion.lower().strip()) for a in annotations]
            emotion_counts = Counter([e[1] for e in emotions])

            # Check if there's a 2-1 split
            if len(emotion_counts) == 2:
                most_common = emotion_counts.most_common()
                if most_common[0][1] == 2 and most_common[1][1] == 1:
                    # Find the minority annotator
                    minority_emotion = most_common[1][0]
                    majority_emotion = most_common[0][0]

                    for ann_name, emotion in emotions:
                        annotator_total_scenarios[ann_name] += 1
                        if emotion == minority_emotion:
                            annotator_minority_count[ann_name] += 1
                            minority_patterns[ann_name].append({
                                "scenario_id": scenario_id,
                                "annotator_emotion": emotion,
                                "majority_emotion": majority_emotion,
                            })

        # Compute annotator statistics
        annotator_stats: Dict[str, Dict[str, Any]] = {}
        outlier_annotators: List[str] = []

        for ann_name, rates in annotator_agreement_rates.items():
            avg_agreement = sum(rates) / len(rates) if rates else 0.0
            total = annotator_total_scenarios.get(ann_name, 0)
            minority = annotator_minority_count.get(ann_name, 0)
            minority_rate = minority / total if total > 0 else 0.0

            stats = {
                "avg_pairwise_agreement": round(avg_agreement, 3),
                "total_2_1_scenarios": total,
                "times_in_minority": minority,
                "minority_rate": round(minority_rate, 3),
                "is_outlier": avg_agreement < threshold or minority_rate > 0.6,
            }
            annotator_stats[ann_name] = stats

            if stats["is_outlier"]:
                outlier_annotators.append(ann_name)

        return {
            "outlier_annotators": outlier_annotators,
            "annotator_stats": annotator_stats,
            "minority_patterns": {k: v[:10] for k, v in minority_patterns.items()},  # Top 10 examples
            "threshold_used": threshold,
        }

    @staticmethod
    def interpret_kappa(kappa: float) -> str:
        """
        Get human-readable interpretation of kappa value.

        Based on Landis & Koch (1977) interpretation scale.
        """
        if kappa < 0.0:
            return "poor (worse than chance)"
        elif kappa < 0.20:
            return "slight"
        elif kappa < 0.40:
            return "fair"
        elif kappa < 0.60:
            return "moderate"
        elif kappa < 0.80:
            return "substantial"
        else:
            return "almost perfect"

    def compute_weighted_kappa_vad(
        self,
        annotations_by_scenario: Dict[int, List[AnnotationData]],
        dimension: str = "valence",
    ) -> float:
        """
        Compute weighted Cohen's kappa for ordinal VAD scales.

        Uses linear weights where disagreement penalty is proportional
        to the distance between ratings on the ordinal scale.

        For multi-rater scenarios (3 annotators), computes average
        pairwise weighted kappa.

        Args:
            annotations_by_scenario: Dict mapping scenario_id -> annotations
            dimension: Which VAD dimension ("valence", "arousal", "dominance")

        Returns:
            Average weighted kappa across all annotator pairs
        """
        # Get ordinal scale values from config
        dim_map = {
            "valence": self.config.valid_labels.valence,
            "arousal": self.config.valid_labels.arousal,
            "dominance": self.config.valid_labels.dominance,
        }

        scale_items = dim_map.get(dimension, [])
        if not scale_items:
            return 0.0

        # Build ordered list of scale values (low to high)
        scale_values = sorted(
            [(item.value.lower(), item.numeric) for item in scale_items], key=lambda x: x[1]
        )
        value_to_idx = {v[0]: i for i, v in enumerate(scale_values)}
        n_categories = len(scale_values)

        if n_categories < 2:
            return 0.0

        # Collect pairwise ratings
        pairwise_kappas: List[float] = []
        annotator_ratings: Dict[str, Dict[int, int]] = defaultdict(dict)

        # Collect ratings per annotator
        for scenario_id, annotations in annotations_by_scenario.items():
            for ann in annotations:
                dim_value = getattr(ann, dimension, "").lower().strip()
                if dim_value in value_to_idx:
                    annotator_ratings[ann.annotator_name][scenario_id] = value_to_idx[dim_value]

        # Compute pairwise weighted kappa for each pair of annotators
        annotators = list(annotator_ratings.keys())

        for ann1, ann2 in combinations(annotators, 2):
            ratings1 = annotator_ratings[ann1]
            ratings2 = annotator_ratings[ann2]

            # Find common scenarios
            common_scenarios = set(ratings1.keys()) & set(ratings2.keys())

            if len(common_scenarios) < 5:
                continue

            # Build rating vectors
            r1 = [ratings1[s] for s in sorted(common_scenarios)]
            r2 = [ratings2[s] for s in sorted(common_scenarios)]

            # Compute weighted kappa with linear weights
            kappa = self._compute_single_weighted_kappa(r1, r2, n_categories)
            pairwise_kappas.append(kappa)

        if not pairwise_kappas:
            return 0.0

        return float(np.mean(pairwise_kappas))

    def _compute_single_weighted_kappa(
        self,
        ratings1: List[int],
        ratings2: List[int],
        n_categories: int,
    ) -> float:
        """
        Compute weighted Cohen's kappa for a single pair of raters.

        Uses linear weights: w_ij = 1 - |i - j| / (k - 1)
        where k is the number of categories.
        """
        n = len(ratings1)
        if n == 0:
            return 0.0

        # Build weight matrix (linear weights)
        weights = np.zeros((n_categories, n_categories))
        for i in range(n_categories):
            for j in range(n_categories):
                weights[i, j] = 1.0 - abs(i - j) / (n_categories - 1)

        # Build observed confusion matrix
        observed = np.zeros((n_categories, n_categories))
        for r1, r2 in zip(ratings1, ratings2):
            observed[r1, r2] += 1
        observed /= n

        # Build expected confusion matrix (marginal products)
        marginals1 = np.sum(observed, axis=1)
        marginals2 = np.sum(observed, axis=0)
        expected = np.outer(marginals1, marginals2)

        # Compute weighted agreement
        p_o = np.sum(weights * observed)
        p_e = np.sum(weights * expected)

        if p_e >= 1.0:
            return 1.0

        return (p_o - p_e) / (1.0 - p_e)

    def compute_confusion_matrix(
        self,
        annotations_by_scenario: Dict[int, List[AnnotationData]],
    ) -> Dict[str, Any]:
        """
        Generate emotion confusion matrix across all annotator pairs.

        This reveals systematic disagreement patterns, e.g., annotators
        frequently confusing sadness with fear.

        Returns:
            Dict containing:
            - matrix: 2D dict of (emotion_i, emotion_j) -> count
            - emotions: List of emotion labels in order
            - total_pairs: Total number of pairwise comparisons
            - agreement_rate: Overall agreement rate
            - common_confusions: List of most frequent confusions
        """
        valid_emotions = sorted([e.lower() for e in self.config.valid_labels.plutchik_emotions])
        emotion_to_idx = {e: i for i, e in enumerate(valid_emotions)}
        n_emotions = len(valid_emotions)

        # Initialize confusion matrix
        matrix = np.zeros((n_emotions, n_emotions), dtype=int)

        # Count pairwise agreements/disagreements
        for scenario_id, annotations in annotations_by_scenario.items():
            emotions = [
                a.plutchik_emotion.lower().strip()
                for a in annotations
                if a.plutchik_emotion and a.plutchik_emotion.lower().strip() in emotion_to_idx
            ]

            # Compare all pairs
            for i, e1 in enumerate(emotions):
                for e2 in emotions[i + 1 :]:
                    idx1 = emotion_to_idx[e1]
                    idx2 = emotion_to_idx[e2]
                    # Count both directions for symmetric matrix
                    matrix[idx1, idx2] += 1
                    if idx1 != idx2:
                        matrix[idx2, idx1] += 1

        # Compute statistics
        total_pairs = int(np.sum(matrix)) // 2  # Avoid double counting diagonal
        diagonal_sum = int(np.trace(matrix))
        agreement_rate = diagonal_sum / max(1, int(np.sum(matrix)))

        # Find most common confusions (off-diagonal)
        confusions: List[Tuple[str, str, int]] = []
        for i in range(n_emotions):
            for j in range(i + 1, n_emotions):
                if matrix[i, j] > 0:
                    confusions.append((valid_emotions[i], valid_emotions[j], int(matrix[i, j])))

        confusions.sort(key=lambda x: -x[2])

        # Convert matrix to dict format for JSON serialization
        matrix_dict: Dict[str, Dict[str, int]] = {}
        for i, e1 in enumerate(valid_emotions):
            matrix_dict[e1] = {}
            for j, e2 in enumerate(valid_emotions):
                matrix_dict[e1][e2] = int(matrix[i, j])

        return {
            "matrix": matrix_dict,
            "emotions": valid_emotions,
            "total_pairs": total_pairs,
            "agreement_count": diagonal_sum,
            "agreement_rate": round(agreement_rate, 4),
            "common_confusions": [
                {"emotion_1": c[0], "emotion_2": c[1], "count": c[2]} for c in confusions[:10]
            ],
        }

    def compute_fleiss_kappa_with_ci(
        self,
        annotations_by_scenario: Dict[int, List[AnnotationData]],
        n_bootstrap: int = 1000,
        confidence_level: float = 0.95,
    ) -> Dict[str, float]:
        """
        Compute Fleiss' kappa with bootstrap confidence intervals.

        Args:
            annotations_by_scenario: Dict mapping scenario_id -> annotations
            n_bootstrap: Number of bootstrap samples
            confidence_level: Confidence level for interval (default 95%)

        Returns:
            Dict with 'kappa', 'ci_lower', 'ci_upper', 'se' (standard error)
        """
        # Compute point estimate
        kappa = self.compute_fleiss_kappa(annotations_by_scenario)

        # Bootstrap for confidence interval
        scenario_ids = list(annotations_by_scenario.keys())
        n_scenarios = len(scenario_ids)

        if n_scenarios < 10:
            # Too few scenarios for reliable bootstrap
            return {
                "kappa": kappa,
                "ci_lower": kappa,
                "ci_upper": kappa,
                "se": 0.0,
                "n_bootstrap": 0,
            }

        np.random.seed(42)  # Reproducibility
        bootstrap_kappas: List[float] = []

        for _ in range(n_bootstrap):
            # Sample scenarios with replacement
            sampled_ids = np.random.choice(scenario_ids, size=n_scenarios, replace=True)
            sampled_annotations = {
                sid: annotations_by_scenario[sid]
                for sid in sampled_ids
                if sid in annotations_by_scenario
            }

            if len(sampled_annotations) >= 5:
                boot_kappa = self.compute_fleiss_kappa(sampled_annotations)
                bootstrap_kappas.append(boot_kappa)

        if not bootstrap_kappas:
            return {
                "kappa": kappa,
                "ci_lower": kappa,
                "ci_upper": kappa,
                "se": 0.0,
                "n_bootstrap": 0,
            }

        # Compute percentile confidence interval
        alpha = 1 - confidence_level
        ci_lower = float(np.percentile(bootstrap_kappas, 100 * alpha / 2))
        ci_upper = float(np.percentile(bootstrap_kappas, 100 * (1 - alpha / 2)))
        se = float(np.std(bootstrap_kappas))

        return {
            "kappa": kappa,
            "ci_lower": round(ci_lower, 4),
            "ci_upper": round(ci_upper, 4),
            "se": round(se, 4),
            "n_bootstrap": len(bootstrap_kappas),
        }
