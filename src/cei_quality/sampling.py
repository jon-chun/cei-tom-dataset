"""
Human Review Sampling for CEI Pipeline.

Generates stratified sampling plans for expert human review.

Sampling strategy:
1. MANDATORY: All critical issues and no-majority cases
2. STRATIFIED: Sample from each (subtype × quality_bucket) stratum
3. ANNOTATOR: Ensure coverage across all annotators
4. FILE REVIEW: Flag files with systematic issues
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from typing import Any, Dict, List, Set, TYPE_CHECKING

from cei_quality.models import (
    AnnotatorQualityReport,
    FileQualityReport,
    RecordQualityReport,
    SamplingPlan,
)
from cei_quality.validators.agreement import InterAnnotatorValidator

if TYPE_CHECKING:
    from cei_quality.config import CEIConfig


class HumanReviewSampler:
    """
    Generate sampling plans for expert human review.

    Uses stratified sampling to ensure coverage across:
    - All 5 pragmatic subtypes
    - All quality levels (excellent to poor)

    Example:
        >>> sampler = HumanReviewSampler(config)
        >>> plan = sampler.generate_plan(
        ...     quality_reports,
        ...     file_reports,
        ...     fleiss_kappa_by_subtype
        ... )
        >>> print(sampler.generate_text_report(plan))
    """

    def __init__(self, config: CEIConfig) -> None:
        """
        Initialize sampler.

        Args:
            config: CEI configuration instance
        """
        self.config = config
        sampling = config.sampling

        self.random_seed = sampling.random_seed
        self.mandatory_priority_threshold = sampling.mandatory_priority_threshold
        self.stratified_rate = sampling.stratified_rate
        self.min_per_subtype = sampling.min_per_subtype

    def generate_plan(
        self,
        quality_reports: List[RecordQualityReport],
        file_reports: List[FileQualityReport],
        fleiss_kappa_by_subtype: Dict[str, float],
    ) -> SamplingPlan:
        """
        Generate comprehensive sampling plan.

        Args:
            quality_reports: Quality reports for all scenarios
            file_reports: Quality reports for all files
            fleiss_kappa_by_subtype: Fleiss' kappa per subtype

        Returns:
            Complete sampling plan
        """
        random.seed(self.random_seed)

        # 1. Mandatory reviews (high priority issues)
        mandatory = [
            r
            for r in quality_reports
            if r.needs_review and r.review_priority >= self.mandatory_priority_threshold
        ]
        mandatory = sorted(mandatory, key=lambda x: (-x.review_priority, x.quality_score))

        selected_ids = {r.scenario_id for r in mandatory}

        # 2. Stratified sampling by (subtype, quality_bucket)
        buckets: Dict[tuple[str, str], List[RecordQualityReport]] = defaultdict(list)

        for report in quality_reports:
            if report.scenario_id in selected_ids:
                continue

            bucket = report.quality_bucket
            buckets[(report.subtype, bucket)].append(report)

        stratified: List[RecordQualityReport] = []

        for (subtype, bucket), bucket_reports in buckets.items():
            # Sample more from lower quality buckets
            if bucket in ["poor", "questionable"]:
                sample_rate = self.stratified_rate * 2
                min_samples = self.min_per_subtype
            elif bucket == "acceptable":
                sample_rate = self.stratified_rate
                min_samples = self.min_per_subtype
            else:
                sample_rate = self.stratified_rate / 2
                min_samples = 1

            n_sample = max(min_samples, int(len(bucket_reports) * sample_rate))
            n_sample = min(n_sample, len(bucket_reports))

            if n_sample > 0:
                sampled = random.sample(bucket_reports, n_sample)
                stratified.extend(sampled)
                selected_ids.update(r.scenario_id for r in sampled)

        # 3. Files needing full review
        files_to_review = [f for f in file_reports if f.needs_full_review]

        # 4. Compute statistics
        all_sampled = mandatory + stratified
        coverage = Counter(r.subtype for r in all_sampled)

        # Compute overall kappa
        if fleiss_kappa_by_subtype:
            overall_kappa = sum(fleiss_kappa_by_subtype.values()) / len(fleiss_kappa_by_subtype)
        else:
            overall_kappa = 0.0

        return SamplingPlan(
            mandatory_reviews=mandatory,
            stratified_sample=stratified,
            files_requiring_review=files_to_review,
            total_scenarios=len(quality_reports),
            scenarios_flagged=len([r for r in quality_reports if r.needs_review]),
            sample_size=len(all_sampled),
            coverage_by_subtype=dict(coverage),
            fleiss_kappa_by_subtype=fleiss_kappa_by_subtype,
            overall_fleiss_kappa=overall_kappa,
        )

    def generate_text_report(self, plan: SamplingPlan) -> str:
        """
        Generate human-readable text report.

        Args:
            plan: Sampling plan to report on

        Returns:
            Formatted text report
        """
        lines = [
            "=" * 70,
            "CEI DATA QUALITY: HUMAN REVIEW SAMPLING PLAN",
            "=" * 70,
            "",
            "SUMMARY",
            "-" * 40,
            f"Total Scenarios:        {plan.total_scenarios}",
            f"Scenarios with Issues:  {plan.scenarios_flagged} "
            f"({100 * plan.scenarios_flagged / max(1, plan.total_scenarios):.1f}%)",
            f"Selected for Review:    {plan.sample_size} "
            f"({100 * plan.sample_size / max(1, plan.total_scenarios):.1f}%)",
            f"  - Mandatory:          {plan.mandatory_count}",
            f"  - Stratified Sample:  {plan.stratified_count}",
            f"Files Needing Review:   {len(plan.files_requiring_review)}",
            "",
            "INTER-ANNOTATOR AGREEMENT (Fleiss' κ)",
            "-" * 40,
            f"Overall: {plan.overall_fleiss_kappa:.3f} "
            f"({InterAnnotatorValidator.interpret_kappa(plan.overall_fleiss_kappa)})",
        ]

        for subtype, kappa in sorted(plan.fleiss_kappa_by_subtype.items()):
            interpretation = InterAnnotatorValidator.interpret_kappa(kappa)
            lines.append(f"  {subtype:30s}: {kappa:.3f} ({interpretation})")

        lines.extend(
            [
                "",
                "COVERAGE BY SUBTYPE",
                "-" * 40,
            ]
        )

        for subtype, count in sorted(plan.coverage_by_subtype.items()):
            lines.append(f"  {subtype:30s}: {count} scenarios")

        # Show top mandatory items
        if plan.mandatory_reviews:
            lines.extend(
                [
                    "",
                    "MANDATORY REVIEW ITEMS (showing top 20)",
                    "-" * 40,
                ]
            )

            for report in plan.mandatory_reviews[:20]:
                lines.append(
                    f"  [{report.subtype}] Scenario {report.scenario_id:3d} | "
                    f"Score: {report.quality_score:.2f} | "
                    f"Priority: {report.review_priority}"
                )
                for reason in report.review_reasons[:2]:
                    lines.append(f"      → {reason}")

        # Show files needing review
        if plan.files_requiring_review:
            lines.extend(
                [
                    "",
                    "FILES REQUIRING FULL REVIEW",
                    "-" * 40,
                ]
            )

            for f in plan.files_requiring_review:
                lines.append(f"  {f.annotator_name} ({f.subtype})")
                lines.append(f"      Reason: {f.review_reason}")

        lines.extend(
            [
                "",
                "=" * 70,
                "END OF REPORT",
                "=" * 70,
            ]
        )

        return "\n".join(lines)

    def ensure_annotator_coverage(
        self,
        quality_reports: List[RecordQualityReport],
        selected_ids: Set[int],
        min_per_annotator: int = 2,
    ) -> List[RecordQualityReport]:
        """
        Ensure minimum representation from each annotator in the sample.

        This catches annotator-specific issues that might be missed if
        one annotator's work is under-sampled.

        Args:
            quality_reports: All quality reports
            selected_ids: Already selected scenario IDs
            min_per_annotator: Minimum scenarios per annotator

        Returns:
            Additional reports to include for annotator coverage
        """
        # Group reports by annotator
        by_annotator: Dict[str, List[RecordQualityReport]] = defaultdict(list)

        for report in quality_reports:
            if report.scenario_id in selected_ids:
                continue
            # Extract annotator from annotations_summary
            annotators = report.annotations_summary.get("annotators", [])
            for annotator in annotators:
                by_annotator[annotator].append(report)

        # Check coverage
        additional: List[RecordQualityReport] = []

        for annotator, reports in by_annotator.items():
            # Count already selected for this annotator
            already_selected = sum(
                1
                for r in quality_reports
                if r.scenario_id in selected_ids
                and annotator in r.annotations_summary.get("annotators", [])
            )

            # Sample more if needed
            need = max(0, min_per_annotator - already_selected)
            if need > 0 and reports:
                # Prioritize lower quality scores
                sorted_reports = sorted(reports, key=lambda x: x.quality_score)
                to_add = sorted_reports[:need]
                additional.extend(to_add)
                selected_ids.update(r.scenario_id for r in to_add)

        return additional

    def generate_annotator_reports(
        self,
        file_reports: List[FileQualityReport],
        quality_reports: List[RecordQualityReport],
    ) -> List[AnnotatorQualityReport]:
        """
        Generate aggregate quality reports per annotator across all subtypes.

        Args:
            file_reports: Per-file quality reports
            quality_reports: Per-scenario quality reports

        Returns:
            List of per-annotator aggregate reports
        """
        import numpy as np

        # Group file reports by annotator
        by_annotator: Dict[str, List[FileQualityReport]] = defaultdict(list)
        for fr in file_reports:
            by_annotator[fr.annotator_name].append(fr)

        annotator_reports: List[AnnotatorQualityReport] = []

        for annotator_name, files in by_annotator.items():
            report = AnnotatorQualityReport(annotator_name=annotator_name)

            # Aggregate from file reports
            report.subtypes_annotated = [f.subtype for f in files]
            report.total_annotations = sum(f.record_count for f in files)

            # Quality scores
            quality_scores = [f.mean_quality_score for f in files if f.mean_quality_score > 0]
            if quality_scores:
                report.mean_quality_score = float(np.mean(quality_scores))
                report.min_quality_score = float(np.min(quality_scores))
                report.std_quality_score = float(np.std(quality_scores))

            # Issues
            report.total_issues = sum(f.issue_count for f in files)

            # Timing
            lead_times = [f.mean_lead_time for f in files if f.mean_lead_time > 0]
            if lead_times:
                report.mean_lead_time = float(np.mean(lead_times))
                report.median_lead_time = float(np.median(lead_times))

            fast_counts = sum(f.suspicious_fast_count for f in files)
            report.dwell_outlier_count = fast_counts
            if report.total_annotations > 0:
                report.dwell_outlier_rate = fast_counts / report.total_annotations

            # Behavioral flags
            report.needs_full_review = any(f.needs_full_review for f in files)
            for f in files:
                if f.needs_full_review:
                    report.review_reasons.append(f"[{f.subtype}] {f.review_reason}")

            # Per-subtype breakdown
            for f in files:
                report.by_subtype[f.subtype] = {
                    "record_count": f.record_count,
                    "issue_count": f.issue_count,
                    "mean_quality_score": round(f.mean_quality_score, 4),
                    "mean_lead_time": round(f.mean_lead_time, 2),
                    "suspicious_fast_count": f.suspicious_fast_count,
                }

            annotator_reports.append(report)

        return annotator_reports
