"""
Stage 1B: Within-File Consistency Validation

Validates consistency within a single annotator's file.

Checks performed:
- No duplicate scenario IDs
- All expected scenario IDs present (1-60)
- Label values are in valid vocabulary
- Lead times are reasonable (not too fast or slow)
- Timestamps are logical
- Intra-annotator consistency (straight-lining, self-contradiction)
- MAD-based dwell-time outlier detection
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

from cei_quality.models import (
    AnnotatorFile,
    FileQualityReport,
    QualityFlag,
    QualityIssue,
)

if TYPE_CHECKING:
    from cei_quality.config import CEIConfig

logger = logging.getLogger(__name__)


# Default thresholds for intra-annotator consistency
DEFAULT_STRAIGHTLINE_THRESHOLD = 0.80  # Flag if >80% same emotion
DEFAULT_DWELL_MAD_THRESHOLD = 2.5  # MAD multiplier for outlier detection


class WithinFileValidator:
    """
    Stage 1B: Validate consistency within a single file.

    This validator checks for issues that can be detected by examining
    a single annotator's file in isolation, such as duplicate IDs,
    invalid labels, and suspicious annotation times.

    Example:
        >>> validator = WithinFileValidator(config)
        >>> issues, file_report = validator.validate_file(annotator_file)
        >>> print(f"Found {file_report.issue_count} issues")
    """

    def __init__(self, config: CEIConfig) -> None:
        """
        Initialize consistency validator.

        Args:
            config: CEI configuration instance
        """
        self.config = config
        self._ls_fields = config.labelstudio
        self._schema = config.schema_config
        self._quality = config.quality

    def validate_file(
        self, af: AnnotatorFile
    ) -> Tuple[Dict[int, List[QualityIssue]], FileQualityReport]:
        """
        Validate all records in an annotator's file.

        Args:
            af: AnnotatorFile containing records to validate

        Returns:
            Tuple of (issues_by_scenario_id, file_report)
        """
        issues_by_scenario: Dict[int, List[QualityIssue]] = defaultdict(list)

        # Initialize file report
        report = FileQualityReport(
            file_path=str(af.file_path),
            annotator_name=af.annotator_name,
            subtype=af.subtype.value,
            record_count=len(af.records),
        )

        # Check record count
        expected_count = self._schema.scenarios_per_subtype
        if len(af.records) != expected_count:
            severity = "critical" if len(af.records) == 0 else "major"
            issues_by_scenario[0].append(
                QualityIssue(
                    flag=QualityFlag.UNEXPECTED_SCENARIO_COUNT,
                    severity=severity,
                    message=f"Expected {expected_count} records, found {len(af.records)}",
                    details={
                        "expected": expected_count,
                        "actual": len(af.records),
                        "file": str(af.file_path),
                    },
                )
            )

        # Track seen scenario IDs for duplicate detection
        seen_scenario_ids: Dict[int, int] = {}  # scenario_id -> first record index
        lead_times: List[Tuple[int, float]] = []

        # Process each record
        for idx, record in enumerate(af.records):
            data = record.get("data", {})
            scenario_id = self._get_scenario_id(data)

            if scenario_id is None:
                continue

            # Check for duplicates
            if scenario_id in seen_scenario_ids:
                issues_by_scenario[scenario_id].append(
                    QualityIssue(
                        flag=QualityFlag.DUPLICATE_SCENARIO_ID,
                        severity="critical",
                        message=f"Duplicate scenario ID: {scenario_id}",
                        details={
                            "scenario_id": scenario_id,
                            "first_index": seen_scenario_ids[scenario_id],
                            "duplicate_index": idx,
                        },
                    )
                )
                report.duplicate_scenarios.append(scenario_id)
            else:
                seen_scenario_ids[scenario_id] = idx

            # Validate label values
            label_issues = self._validate_label_values(record, scenario_id)
            issues_by_scenario[scenario_id].extend(label_issues)

            # Collect lead times
            lead_time = self._extract_lead_time(record)
            if lead_time is not None:
                lead_times.append((scenario_id, lead_time))

            # Check timestamps
            timestamp_issues = self._validate_timestamps(record, scenario_id)
            issues_by_scenario[scenario_id].extend(timestamp_issues)

        # Check for missing scenario IDs
        # NOTE: We no longer assume IDs must be 1-N contiguous.
        # If record count matches expected count, scenarios are considered complete.
        # Missing scenarios are detected at the cross-annotation level (Stage 1C)
        # where we compare across annotators for the same subtype.
        found_ids = set(seen_scenario_ids.keys())

        # Only flag if we have fewer records than expected AND some expected IDs are missing
        if len(af.records) < expected_count:
            # We don't know the "expected" IDs without cross-referencing other annotators
            # This check is now informational only - actual validation happens in Stage 1C
            report.missing_scenarios = []  # Will be populated by cross-annotator check

        # Analyze lead times for anomalies
        if lead_times:
            lead_time_issues = self._detect_lead_time_anomalies(lead_times)
            for scenario_id, issue in lead_time_issues:
                issues_by_scenario[scenario_id].append(issue)
                if "fast" in issue.details.get("type", ""):
                    report.suspicious_fast_count += 1
                else:
                    report.suspicious_slow_count += 1

            # Compute timing statistics
            all_times = [lt for _, lt in lead_times]
            report.mean_lead_time = float(np.mean(all_times))
            report.min_lead_time = float(np.min(all_times))
            report.max_lead_time = float(np.max(all_times))

        # Update report with issue counts
        total_issues = sum(len(issues) for issues in issues_by_scenario.values())
        report.issue_count = total_issues

        # Determine if file needs full review
        critical_issues = sum(
            1
            for issues in issues_by_scenario.values()
            for issue in issues
            if issue.severity == "critical"
        )

        threshold = (
            self._quality.sampling.file_issue_rate_threshold
            if hasattr(self._quality, "sampling")
            else 0.20
        )
        issue_rate = total_issues / max(1, len(af.records))

        if critical_issues > 5 or len(report.missing_scenarios) > 3:
            report.needs_full_review = True
            report.review_reason = f"High critical issues ({critical_issues}) or missing scenarios ({len(report.missing_scenarios)})"
        elif issue_rate > 0.20:
            report.needs_full_review = True
            report.review_reason = f"High issue rate ({issue_rate:.1%})"
        elif report.suspicious_fast_count > 10:
            report.needs_full_review = True
            report.review_reason = (
                f"Many suspiciously fast annotations ({report.suspicious_fast_count})"
            )

        return dict(issues_by_scenario), report

    def _get_scenario_id(self, data: Dict[str, Any]) -> Optional[int]:
        """Extract scenario ID from data dict."""
        field = self._ls_fields.data_fields.scenario_id
        value = data.get(field)

        if value is None:
            return None

        if isinstance(value, int):
            return value

        try:
            return int(value)
        except (ValueError, TypeError):
            return None

    def _validate_label_values(
        self, record: Dict[str, Any], scenario_id: int
    ) -> List[QualityIssue]:
        """Validate that label values are in valid vocabulary."""
        issues: List[QualityIssue] = []

        annotations = record.get("annotations", [])
        if not annotations:
            return issues

        # Check first annotation's results
        result = annotations[0].get("result", [])

        for item in result:
            from_name = item.get("from_name", "")
            choices = item.get("value", {}).get("choices", [])

            if not choices:
                continue

            value = choices[0]

            # Determine which field this is and validate
            issues.extend(self._check_label_value(from_name, value, scenario_id))

        return issues

    def _check_label_value(
        self, field_name: str, value: str, scenario_id: int
    ) -> List[QualityIssue]:
        """Check if a label value is valid."""
        issues: List[QualityIssue] = []

        # Map field names to validation sets
        label_fields = self._ls_fields.label_fields
        valid_labels = self.config.valid_labels

        field_to_valid = {
            label_fields.plutchik_emotion: valid_labels.plutchik_emotions,
            label_fields.valence: [v.value for v in valid_labels.valence]
            if valid_labels.valence
            else None,
            label_fields.arousal: [v.value for v in valid_labels.arousal]
            if valid_labels.arousal
            else None,
            label_fields.dominance: [v.value for v in valid_labels.dominance]
            if valid_labels.dominance
            else None,
            label_fields.confidence: [v.value for v in valid_labels.confidence]
            if valid_labels.confidence
            else None,
        }

        valid_values = field_to_valid.get(field_name)

        if valid_values is None:
            # Unknown field, skip validation
            return issues

        # Normalize for comparison
        value_lower = value.lower().strip()
        valid_lower = {v.lower() for v in valid_values}

        # Check exact match or partial match
        is_valid = value_lower in valid_lower or any(
            value_lower in v or v in value_lower for v in valid_lower
        )

        if not is_valid:
            issues.append(
                QualityIssue(
                    flag=QualityFlag.INVALID_LABEL_VALUE,
                    severity="major",
                    message=f"Unexpected value '{value}' for field '{field_name}'",
                    details={
                        "scenario_id": scenario_id,
                        "field": field_name,
                        "value": value,
                        "expected_examples": list(valid_values)[:5],
                    },
                )
            )

        return issues

    def _extract_lead_time(self, record: Dict[str, Any]) -> Optional[float]:
        """Extract lead time from annotation."""
        annotations = record.get("annotations", [])
        if not annotations:
            return None

        lead_time = annotations[0].get("lead_time")

        if lead_time is not None:
            try:
                return float(lead_time)
            except (ValueError, TypeError):
                return None

        return None

    def _validate_timestamps(self, record: Dict[str, Any], scenario_id: int) -> List[QualityIssue]:
        """Check for timestamp anomalies."""
        issues: List[QualityIssue] = []

        annotations = record.get("annotations", [])
        if not annotations:
            return issues

        ann = annotations[0]
        created_at = ann.get("created_at")
        updated_at = ann.get("updated_at")
        draft_created_at = ann.get("draft_created_at")

        # Check if updated before created
        if created_at and updated_at:
            if updated_at < created_at:
                issues.append(
                    QualityIssue(
                        flag=QualityFlag.TIMESTAMP_ANOMALY,
                        severity="minor",
                        message="Updated timestamp before created timestamp",
                        details={
                            "scenario_id": scenario_id,
                            "created_at": created_at,
                            "updated_at": updated_at,
                        },
                    )
                )

        return issues

    def _detect_lead_time_anomalies(
        self, lead_times: List[Tuple[int, float]]
    ) -> List[Tuple[int, QualityIssue]]:
        """
        Detect anomalous annotation times.

        Uses configurable thresholds:
        - impossibly_fast: Critical (likely random clicking)
        - suspiciously_fast: Major (needs verification)
        - unusually_slow: Info (possible distraction)
        """
        if len(lead_times) < 5:
            return []

        # Get thresholds from config
        lt_config = self.config.quality.lead_time
        impossibly_fast = lt_config.impossibly_fast
        suspiciously_fast = lt_config.suspiciously_fast
        unusually_slow = lt_config.unusually_slow

        anomalies: List[Tuple[int, QualityIssue]] = []

        for scenario_id, lead_time in lead_times:
            if lead_time < impossibly_fast:
                anomalies.append(
                    (
                        scenario_id,
                        QualityIssue(
                            flag=QualityFlag.SUSPICIOUS_LEAD_TIME,
                            severity="critical",
                            message=f"Impossibly fast annotation: {lead_time:.1f}s",
                            details={
                                "scenario_id": scenario_id,
                                "lead_time": lead_time,
                                "threshold": impossibly_fast,
                                "type": "impossibly_fast",
                            },
                        ),
                    )
                )
            elif lead_time < suspiciously_fast:
                anomalies.append(
                    (
                        scenario_id,
                        QualityIssue(
                            flag=QualityFlag.SUSPICIOUS_LEAD_TIME,
                            severity="major",
                            message=f"Suspiciously fast annotation: {lead_time:.1f}s",
                            details={
                                "scenario_id": scenario_id,
                                "lead_time": lead_time,
                                "threshold": suspiciously_fast,
                                "type": "suspiciously_fast",
                            },
                        ),
                    )
                )
            elif lead_time > unusually_slow:
                anomalies.append(
                    (
                        scenario_id,
                        QualityIssue(
                            flag=QualityFlag.SUSPICIOUS_LEAD_TIME,
                            severity="info",
                            message=f"Unusually slow annotation: {lead_time:.0f}s ({lead_time / 60:.1f} min)",
                            details={
                                "scenario_id": scenario_id,
                                "lead_time": lead_time,
                                "threshold": unusually_slow,
                                "type": "unusually_slow",
                            },
                        ),
                    )
                )

        return anomalies

    def detect_straight_lining(
        self,
        af: AnnotatorFile,
        threshold: float = DEFAULT_STRAIGHTLINE_THRESHOLD,
    ) -> Optional[QualityIssue]:
        """
        Detect if annotator uses the same emotion for too many scenarios.

        Straight-lining indicates the annotator may not be carefully
        reading each scenario.

        Args:
            af: AnnotatorFile containing records to analyze
            threshold: Proportion threshold (default 0.80 = 80%)

        Returns:
            QualityIssue if straight-lining detected, None otherwise
        """
        emotions: List[str] = []

        for record in af.records:
            annotations = record.get("annotations", [])
            if not annotations:
                continue

            result = annotations[0].get("result", [])
            label_field = self._ls_fields.label_fields.plutchik_emotion

            for item in result:
                if item.get("from_name") == label_field:
                    choices = item.get("value", {}).get("choices", [])
                    if choices:
                        emotions.append(choices[0].lower().strip())
                        break

        if len(emotions) < 10:
            return None

        emotion_counts = Counter(emotions)
        most_common_emotion, most_common_count = emotion_counts.most_common(1)[0]
        proportion = most_common_count / len(emotions)

        if proportion > threshold:
            return QualityIssue(
                flag=QualityFlag.STRAIGHT_LINING,
                severity="major",
                message=f"Possible straight-lining: {most_common_emotion} used in {proportion:.0%} of annotations",
                details={
                    "annotator": af.annotator_name,
                    "subtype": af.subtype.value,
                    "dominant_emotion": most_common_emotion,
                    "proportion": round(proportion, 3),
                    "count": most_common_count,
                    "total": len(emotions),
                    "threshold": threshold,
                    "emotion_distribution": dict(emotion_counts),
                },
            )

        return None

    def detect_dwell_time_outliers_mad(
        self,
        lead_times: List[Tuple[int, float]],
        mad_threshold: float = DEFAULT_DWELL_MAD_THRESHOLD,
    ) -> Tuple[List[Tuple[int, QualityIssue]], Dict[str, Any]]:
        """
        Detect dwell-time outliers using Median Absolute Deviation (MAD).

        MAD is more robust to outliers than standard deviation, making it
        better suited for detecting suspicious annotation times.

        Modified Z-score = 0.6745 * (x - median) / MAD
        Outliers are those with |modified_z| > threshold

        Args:
            lead_times: List of (scenario_id, lead_time) tuples
            mad_threshold: Number of MADs from median to flag (default 2.5)

        Returns:
            Tuple of (list of issues, statistics dict)
        """
        if len(lead_times) < 5:
            return [], {}

        times = np.array([lt for _, lt in lead_times])
        median_time = float(np.median(times))

        # Compute MAD (Median Absolute Deviation)
        mad = float(np.median(np.abs(times - median_time)))

        # Handle zero MAD (all same values)
        if mad < 0.001:
            mad = float(np.std(times)) / 1.4826  # Fall back to scaled SD

        if mad < 0.001:
            return [], {"median": median_time, "mad": 0.0, "outlier_count": 0}

        # Compute modified Z-scores
        # 0.6745 is the scaling factor for consistency with normal distribution
        modified_z_scores = 0.6745 * (times - median_time) / mad

        outliers: List[Tuple[int, QualityIssue]] = []
        outlier_count = 0

        for i, (scenario_id, lead_time) in enumerate(lead_times):
            z_score = modified_z_scores[i]

            if abs(z_score) > mad_threshold:
                outlier_count += 1
                severity = "major" if abs(z_score) > mad_threshold * 1.5 else "minor"
                direction = "fast" if z_score < 0 else "slow"

                outliers.append(
                    (
                        scenario_id,
                        QualityIssue(
                            flag=QualityFlag.DWELL_TIME_OUTLIER,
                            severity=severity,
                            message=f"Dwell time outlier ({direction}): {lead_time:.1f}s (z={z_score:.2f})",
                            details={
                                "scenario_id": scenario_id,
                                "lead_time": lead_time,
                                "median_time": round(median_time, 2),
                                "mad": round(mad, 2),
                                "modified_z_score": round(z_score, 2),
                                "threshold": mad_threshold,
                                "direction": direction,
                            },
                        ),
                    )
                )

        stats = {
            "median": round(median_time, 2),
            "mad": round(mad, 2),
            "outlier_count": outlier_count,
            "outlier_rate": round(outlier_count / len(lead_times), 3),
            "threshold": mad_threshold,
        }

        return outliers, stats

    def detect_annotator_aggregate_dwell_outlier(
        self,
        af: AnnotatorFile,
        all_annotator_stats: Dict[str, Dict[str, float]],
        mad_threshold: float = DEFAULT_DWELL_MAD_THRESHOLD,
    ) -> Optional[QualityIssue]:
        """
        Detect if an annotator's average dwell time is an outlier
        compared to all annotators in the cohort.

        This catches annotators who are consistently too fast or slow
        across their entire annotation session.

        Args:
            af: AnnotatorFile to check
            all_annotator_stats: Dict mapping annotator_name -> timing stats
            mad_threshold: MAD multiplier threshold

        Returns:
            QualityIssue if annotator is outlier, None otherwise
        """
        if len(all_annotator_stats) < 3:
            return None

        # Get this annotator's median
        this_stats = all_annotator_stats.get(af.annotator_name)
        if not this_stats or "median" not in this_stats:
            return None

        this_median = this_stats["median"]

        # Get all annotator medians
        all_medians = [
            stats["median"] for stats in all_annotator_stats.values() if "median" in stats
        ]

        if len(all_medians) < 3:
            return None

        all_medians_arr = np.array(all_medians)
        cohort_median = float(np.median(all_medians_arr))
        cohort_mad = float(np.median(np.abs(all_medians_arr - cohort_median)))

        if cohort_mad < 0.01:
            return None

        z_score = 0.6745 * (this_median - cohort_median) / cohort_mad

        if abs(z_score) > mad_threshold:
            direction = "faster" if z_score < 0 else "slower"
            return QualityIssue(
                flag=QualityFlag.ANNOTATOR_DWELL_OUTLIER,
                severity="major",
                message=f"Annotator {af.annotator_name} is consistently {direction} than cohort (z={z_score:.2f})",
                details={
                    "annotator": af.annotator_name,
                    "subtype": af.subtype.value,
                    "annotator_median": round(this_median, 2),
                    "cohort_median": round(cohort_median, 2),
                    "cohort_mad": round(cohort_mad, 2),
                    "modified_z_score": round(z_score, 2),
                    "threshold": mad_threshold,
                    "direction": direction,
                },
            )

        return None

    def detect_self_contradictions(
        self,
        af: AnnotatorFile,
    ) -> List[QualityIssue]:
        """
        Detect intra-annotator contradictions where the same annotator
        gives inconsistent VAD-emotion mappings.

        Example: If annotator labels "joy" with "very unpleasant" valence
        multiple times, this indicates confusion or careless annotation.

        Args:
            af: AnnotatorFile to analyze

        Returns:
            List of self-contradiction issues
        """
        issues: List[QualityIssue] = []

        # Track emotion -> valence mappings for this annotator
        emotion_valence_map: Dict[str, List[str]] = defaultdict(list)

        for record in af.records:
            annotations = record.get("annotations", [])
            if not annotations:
                continue

            result = annotations[0].get("result", [])

            emotion = None
            valence = None

            label_fields = self._ls_fields.label_fields

            for item in result:
                from_name = item.get("from_name")
                choices = item.get("value", {}).get("choices", [])

                if from_name == label_fields.plutchik_emotion and choices:
                    emotion = choices[0].lower().strip()
                elif from_name == label_fields.valence and choices:
                    valence = choices[0].lower().strip()

            if emotion and valence:
                emotion_valence_map[emotion].append(valence)

        # Check for contradictions: same emotion with opposite valences
        positive_valences = {"very pleasant", "pleasant", "slightly pleasant"}
        negative_valences = {"very unpleasant", "unpleasant", "slightly unpleasant"}

        for emotion, valences in emotion_valence_map.items():
            if len(valences) < 2:
                continue

            has_positive = any(v in positive_valences for v in valences)
            has_negative = any(v in negative_valences for v in valences)

            if has_positive and has_negative:
                valence_counts = Counter(valences)
                issues.append(
                    QualityIssue(
                        flag=QualityFlag.SELF_CONTRADICTION,
                        severity="minor",
                        message=f"Annotator gave both positive and negative valence for '{emotion}'",
                        details={
                            "annotator": af.annotator_name,
                            "emotion": emotion,
                            "valence_distribution": dict(valence_counts),
                            "has_positive": has_positive,
                            "has_negative": has_negative,
                        },
                    )
                )

        return issues
