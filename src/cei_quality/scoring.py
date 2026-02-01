"""
Quality Score Aggregation for CEI Pipeline.

Combines issues from all validation stages into composite quality scores.

Score computation:
- Each issue has a severity penalty
- Stage scores are weighted and combined
- Final score is 0.0 (worst) to 1.0 (best)
"""

from __future__ import annotations

from typing import Any, Dict, List, TYPE_CHECKING

from cei_quality.models import (
    QualityFlag,
    QualityIssue,
    RecordQualityReport,
)

if TYPE_CHECKING:
    from cei_quality.config import CEIConfig


class QualityScoreAggregator:
    """
    Aggregate quality issues into composite scores.

    Uses configurable weights for:
    - Severity penalties (critical, major, minor, info)
    - Stage weights (schema, consistency, agreement, plausibility)

    Example:
        >>> aggregator = QualityScoreAggregator(config)
        >>> report = aggregator.aggregate_scenario(
        ...     scenario_id=1,
        ...     task_id=1001,
        ...     subtype="sarcasm-irony",
        ...     file_paths=["file1.json", "file2.json"],
        ...     issues_1a=[...],
        ...     issues_1b=[...],
        ...     issues_1c=[...],
        ...     issues_1d=[...]
        ... )
        >>> print(f"Quality score: {report.quality_score:.2f}")
    """

    def __init__(self, config: CEIConfig) -> None:
        """
        Initialize score aggregator.

        Args:
            config: CEI configuration instance
        """
        self.config = config

        # Load weights from config
        sw = config.quality.severity_weights
        self.severity_weights = {
            "critical": sw.critical,
            "major": sw.major,
            "minor": sw.minor,
            "info": sw.info,
        }

        stw = config.quality.stage_weights
        self.stage_weights = {
            "schema": stw.schema,
            "consistency": stw.consistency,
            "agreement": stw.agreement,
            "plausibility": stw.plausibility,
        }

    def aggregate_scenario(
        self,
        scenario_id: int,
        task_id: int,
        subtype: str,
        file_paths: List[str],
        issues_1a: List[QualityIssue],
        issues_1b: List[QualityIssue],
        issues_1c: List[QualityIssue],
        issues_1d: List[QualityIssue],
        scenario_text: str = "",
        annotations_summary: Dict[str, Any] | None = None,
    ) -> RecordQualityReport:
        """
        Aggregate all issues for a scenario into a quality report.

        Args:
            scenario_id: Scenario identifier
            task_id: Label Studio task ID
            subtype: Pragmatic subtype name
            file_paths: Source file paths
            issues_1a: Stage 1A schema issues
            issues_1b: Stage 1B consistency issues
            issues_1c: Stage 1C agreement issues
            issues_1d: Stage 1D plausibility issues
            scenario_text: Formatted scenario text for review
            annotations_summary: Summary of annotations for review

        Returns:
            Complete quality report for the scenario
        """
        report = RecordQualityReport(
            scenario_id=scenario_id,
            task_id=task_id,
            subtype=subtype,
            file_paths=file_paths,
            scenario_text=scenario_text,
            annotations_summary=annotations_summary or {},
        )

        # Collect all issues
        all_issues = issues_1a + issues_1b + issues_1c + issues_1d
        report.issues = all_issues

        # Compute per-stage scores
        report.schema_score = self._compute_stage_score(issues_1a)
        report.consistency_score = self._compute_stage_score(issues_1b)
        report.agreement_score = self._compute_agreement_score(issues_1c)
        report.plausibility_score = self._compute_stage_score(issues_1d)

        # Compute composite score
        report.quality_score = (
            self.stage_weights["schema"] * report.schema_score
            + self.stage_weights["consistency"] * report.consistency_score
            + self.stage_weights["agreement"] * report.agreement_score
            + self.stage_weights["plausibility"] * report.plausibility_score
        )

        # Set review flags
        self._set_review_flags(report, all_issues)

        return report

    def _compute_stage_score(self, issues: List[QualityIssue]) -> float:
        """
        Compute score for a single stage.

        Score = max(0, 1 - sum of severity penalties)
        """
        if not issues:
            return 1.0

        penalty = sum(self.severity_weights.get(issue.severity, 0) for issue in issues)

        return max(0.0, 1.0 - penalty)

    def _compute_agreement_score(self, issues: List[QualityIssue]) -> float:
        """
        Compute agreement score with special handling.

        For complete disagreement (no majority), returns 0.33.
        Otherwise uses standard penalty calculation.
        """
        if not issues:
            return 1.0

        # Check for no-majority flag (worst case for agreement)
        for issue in issues:
            if issue.flag == QualityFlag.NO_MAJORITY_EMOTION:
                return 0.33

        return self._compute_stage_score(issues)

    def _set_review_flags(
        self, report: RecordQualityReport, all_issues: List[QualityIssue]
    ) -> None:
        """Set review needs and priority based on issues."""
        critical_count = sum(1 for i in all_issues if i.severity == "critical")
        major_count = sum(1 for i in all_issues if i.severity == "major")

        sampling_config = self.config.sampling

        # Critical issues always need review
        if critical_count > 0:
            report.needs_review = True
            report.review_priority = 10
            report.review_reasons.append(f"{critical_count} critical issue(s)")

        # Multiple major issues
        if major_count >= 2:
            report.needs_review = True
            report.review_priority = max(report.review_priority, 7)
            report.review_reasons.append(f"{major_count} major issues")

        # Low quality score
        if report.quality_score < sampling_config.mandatory_score_threshold:
            report.needs_review = True
            report.review_priority = max(report.review_priority, 8)
            report.review_reasons.append(f"Low quality score: {report.quality_score:.2f}")

        # Specific issue flags
        for issue in all_issues:
            if issue.flag == QualityFlag.NO_MAJORITY_EMOTION:
                report.needs_review = True
                report.review_priority = max(report.review_priority, 9)
                if "No majority emotion agreement" not in report.review_reasons:
                    report.review_reasons.append("No majority emotion agreement")

            if issue.flag == QualityFlag.IMPLAUSIBLE_LISTENER_EMOTION:
                report.needs_review = True
                report.review_priority = max(report.review_priority, 6)
                if "Implausible listener emotion" not in report.review_reasons:
                    report.review_reasons.append("Implausible listener emotion (LLM)")

            if issue.flag == QualityFlag.SPEAKER_LISTENER_CONFUSION:
                report.needs_review = True
                report.review_priority = max(report.review_priority, 8)
                if "Speaker/listener confusion" not in report.review_reasons:
                    report.review_reasons.append("Possible speaker/listener confusion")
