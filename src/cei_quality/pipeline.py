"""
CEI Quality Pipeline Orchestrator.

Main entry point that coordinates all validation stages
and generates final quality reports.

Pipeline stages:
1. Load and parse annotation files
2. Stage 1A: Schema validation
3. Stage 1B: Within-file consistency (+ intra-annotator checks)
4. Stage 1C: Inter-annotator agreement (+ confusion matrix, weighted kappa)
5. Stage 1D: Plausibility checks (+ LLM corroboration)
6. Score aggregation
7. Sampling plan generation (+ annotator stratification)
8. Output generation (+ run metadata, annotator reports)
"""

from __future__ import annotations

import hashlib
import json
import logging
import platform
import subprocess
import sys
import uuid
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from cei_quality.config import CEIConfig
from cei_quality.loaders import DataLoader
from cei_quality.models import (
    AnnotationData,
    AnnotatorFile,
    AnnotatorQualityReport,
    FileQualityReport,
    PragmaticSubtype,
    QualityIssue,
    RecordQualityReport,
    RunMetadata,
    SamplingPlan,
    ScenarioData,
)
from cei_quality.sampling import HumanReviewSampler
from cei_quality.scoring import QualityScoreAggregator
from cei_quality.validators import (
    InterAnnotatorValidator,
    PlausibilityChecker,
    SchemaValidator,
    WithinFileValidator,
)
from cei_quality.differentiation import DifferentiationCalculator, DifferentiationScore
from cei_quality.review_report import ReviewReportGenerator, generate_qa_review_report
from cei_quality.comprehensive_report import generate_comprehensive_report

logger = logging.getLogger(__name__)


class CEIQualityPipeline:
    """
    Main orchestrator for the CEI quality filtering pipeline.

    Coordinates all validation stages and produces final reports.

    Example:
        >>> from cei_quality import CEIQualityPipeline, load_config
        >>>
        >>> config = load_config("config/config.yml")
        >>> pipeline = CEIQualityPipeline(config)
        >>> sampling_plan = pipeline.run()
        >>>
        >>> print(f"Found {len(pipeline.quality_reports)} scenarios")
        >>> print(f"Selected {sampling_plan.sample_size} for review")
    """

    def __init__(self, config: CEIConfig) -> None:
        """
        Initialize pipeline with configuration.

        Args:
            config: CEI configuration instance
        """
        self.config = config

        # Resolve paths
        self.paths = config.get_resolved_paths()

        # Initialize components
        self.loader = DataLoader(config)
        self.schema_validator = SchemaValidator(config)
        self.consistency_validator = WithinFileValidator(config)
        self.agreement_validator = InterAnnotatorValidator(config)
        self.plausibility_checker = PlausibilityChecker(config)
        self.score_aggregator = QualityScoreAggregator(config)
        self.sampler = HumanReviewSampler(config)

        # Results storage
        self.quality_reports: List[RecordQualityReport] = []
        self.file_reports: List[FileQualityReport] = []
        self.annotator_reports: List[AnnotatorQualityReport] = []
        self.fleiss_kappa_by_subtype: Dict[str, float] = {}

        # Enhanced agreement metrics
        self.kappa_with_ci_by_subtype: Dict[str, Dict[str, float]] = {}
        self.weighted_kappa_by_subtype: Dict[str, Dict[str, float]] = {}
        self.confusion_matrices_by_subtype: Dict[str, Dict[str, Any]] = {}
        self.systematic_outliers_by_subtype: Dict[str, Dict[str, Any]] = {}

        # LLM corroboration summary
        self.llm_corroboration_summary: Dict[str, Any] = {}

        # Run metadata
        self.run_metadata: Optional[RunMetadata] = None

        # Annotator dwell-time statistics (for cohort outlier detection)
        self._annotator_dwell_stats: Dict[str, Dict[str, float]] = {}

        # Data storage
        self._scenarios: Dict[tuple[str, int], ScenarioData] = {}
        self._annotations: Dict[tuple[str, int], List[AnnotationData]] = {}
        self._organized_files: Dict[PragmaticSubtype, Dict[str, AnnotatorFile]] = {}

        # Differentiation scoring
        self.differentiation_calculator = DifferentiationCalculator(config)
        self.differentiation_scores: Dict[int, DifferentiationScore] = {}

    def run(self) -> SamplingPlan:
        """
        Execute the complete quality pipeline.

        Returns:
            Sampling plan for human review
        """
        logger.info("=" * 70)
        logger.info("CEI DATA QUALITY PIPELINE (Enhanced)")
        logger.info("=" * 70)

        # Ensure output directory exists
        self.paths.output_dir.mkdir(parents=True, exist_ok=True)

        # Load all data
        logger.info("\n[Stage 0] Loading annotation files...")
        organized_files = self.loader.load_all()
        self._organized_files = organized_files
        total_files = sum(len(v) for v in organized_files.values())
        total_records = sum(
            len(af.records) for afs in organized_files.values() for af in afs.values()
        )
        logger.info(
            f"  Loaded {total_files} files, {total_records} records across {len(organized_files)} subtypes"
        )

        # Generate run metadata
        self.run_metadata = self._generate_run_metadata(total_files, total_records)
        logger.info(
            f"  Run ID: {self.run_metadata.run_id}, Config hash: {self.run_metadata.config_hash}"
        )

        # Collect issues by stage
        all_issues_1a: Dict[tuple[str, int], List[QualityIssue]] = defaultdict(list)
        all_issues_1b: Dict[tuple[str, int], List[QualityIssue]] = defaultdict(list)
        all_issues_1c: Dict[tuple[str, int], List[QualityIssue]] = defaultdict(list)
        all_issues_1d: Dict[tuple[str, int], List[QualityIssue]] = defaultdict(list)

        # Process each subtype
        for subtype, annotator_files in organized_files.items():
            subtype_name = subtype.value
            logger.info(f"\n[Processing] {subtype_name} ({len(annotator_files)} annotators)")

            subtype_scenarios: Dict[int, ScenarioData] = {}
            subtype_annotations: Dict[int, List[AnnotationData]] = defaultdict(list)
            subtype_lead_times: Dict[str, List[tuple[int, float]]] = defaultdict(list)

            # Stages 1A & 1B: File-level validation
            for annotator_name, af in annotator_files.items():
                logger.info(f"  [1A/1B] Validating {annotator_name}...")

                # Stage 1A: Schema validation
                schema_issues = self.schema_validator.validate_file(af.records, str(af.file_path))
                for scenario_id, issues in schema_issues.items():
                    key = (subtype_name, scenario_id)
                    all_issues_1a[key].extend(issues)

                # Stage 1B: Within-file consistency
                consistency_issues, file_report = self.consistency_validator.validate_file(af)
                self.file_reports.append(file_report)

                for scenario_id, issues in consistency_issues.items():
                    key = (subtype_name, scenario_id)
                    all_issues_1b[key].extend(issues)

                # === NEW: Intra-annotator consistency checks ===

                # Straight-lining detection
                straight_line_issue = self.consistency_validator.detect_straight_lining(af)
                if straight_line_issue:
                    all_issues_1b[(subtype_name, 0)].append(straight_line_issue)
                    file_report.needs_full_review = True
                    file_report.review_reason = straight_line_issue.message

                # Self-contradiction detection
                self_contradiction_issues = self.consistency_validator.detect_self_contradictions(
                    af
                )
                for issue in self_contradiction_issues:
                    all_issues_1b[(subtype_name, 0)].append(issue)

                # Collect lead times for MAD-based outlier detection
                for record in af.records:
                    annotations = record.get("annotations", [])
                    if annotations:
                        lt = annotations[0].get("lead_time")
                        sid = record.get("data", {}).get("id")
                        if lt is not None and sid is not None:
                            subtype_lead_times[annotator_name].append((sid, float(lt)))

                # Parse and store data
                for record in af.records:
                    scenario, annotation, _ = self.loader.parse_record(record, af)

                    if scenario:
                        subtype_scenarios[scenario.scenario_id] = scenario
                        self._scenarios[(subtype_name, scenario.scenario_id)] = scenario

                    if annotation:
                        subtype_annotations[scenario.scenario_id].append(annotation)
                        self._annotations.setdefault(
                            (subtype_name, annotation.scenario_id), []
                        ).append(annotation)

            # === NEW: MAD-based dwell time outlier detection ===
            for annotator_name, lead_times in subtype_lead_times.items():
                mad_outliers, dwell_stats = (
                    self.consistency_validator.detect_dwell_time_outliers_mad(lead_times)
                )
                self._annotator_dwell_stats[f"{subtype_name}_{annotator_name}"] = dwell_stats

                for scenario_id, issue in mad_outliers:
                    key = (subtype_name, scenario_id)
                    all_issues_1b[key].append(issue)

            # Stage 1C: Inter-annotator agreement
            logger.info(f"  [1C] Checking inter-annotator agreement...")
            agreement_issues = self.agreement_validator.validate_subtype(
                subtype,
                annotator_files,
                subtype_scenarios,
                dict(subtype_annotations),
            )

            for scenario_id, issues in agreement_issues.items():
                key = (subtype_name, scenario_id)
                all_issues_1c[key] = issues

            # Compute Fleiss' kappa
            kappa = self.agreement_validator.compute_fleiss_kappa(dict(subtype_annotations))
            self.fleiss_kappa_by_subtype[subtype_name] = kappa
            logger.info(f"    Fleiss' κ = {kappa:.3f}")

            # === NEW: Enhanced agreement metrics ===

            # Kappa with bootstrap confidence intervals
            kappa_with_ci = self.agreement_validator.compute_fleiss_kappa_with_ci(
                dict(subtype_annotations)
            )
            self.kappa_with_ci_by_subtype[subtype_name] = kappa_with_ci
            logger.info(
                f"    95% CI: [{kappa_with_ci['ci_lower']:.3f}, {kappa_with_ci['ci_upper']:.3f}]"
            )

            # Weighted kappa for VAD dimensions
            self.weighted_kappa_by_subtype[subtype_name] = {
                "valence": self.agreement_validator.compute_weighted_kappa_vad(
                    dict(subtype_annotations), "valence"
                ),
                "arousal": self.agreement_validator.compute_weighted_kappa_vad(
                    dict(subtype_annotations), "arousal"
                ),
                "dominance": self.agreement_validator.compute_weighted_kappa_vad(
                    dict(subtype_annotations), "dominance"
                ),
            }

            # Confusion matrix
            confusion_matrix = self.agreement_validator.compute_confusion_matrix(
                dict(subtype_annotations)
            )
            self.confusion_matrices_by_subtype[subtype_name] = confusion_matrix

            # Systematic outlier detection
            systematic_outliers = self.agreement_validator.detect_systematic_outliers(
                dict(subtype_annotations)
            )
            self.systematic_outliers_by_subtype[subtype_name] = systematic_outliers
            if systematic_outliers.get("outlier_annotators"):
                logger.info(
                    f"    Systematic outliers: {systematic_outliers['outlier_annotators']}"
                )

        # Stage 1D: Plausibility checks
        logger.info("\n[Stage 1D] Running plausibility checks...")
        for key, annotations in self._annotations.items():
            subtype_name, scenario_id = key
            scenario = self._scenarios.get(key)

            if scenario:
                subtype = PragmaticSubtype.from_string(subtype_name)
                if subtype:
                    issues = self.plausibility_checker.check_scenario(
                        scenario, annotations, subtype
                    )
                    all_issues_1d[key] = issues

        # Optional LLM checks with corroboration
        if self.config.llm.enabled:
            logger.info("  Running LLM plausibility checks (this may take a while)...")
            scenarios_for_llm = []
            for key, annotations in self._annotations.items():
                subtype_name, scenario_id = key
                scenario = self._scenarios.get(key)
                subtype = PragmaticSubtype.from_string(subtype_name)
                if scenario and subtype:
                    scenarios_for_llm.append((scenario, annotations, subtype))

            raw_llm_issues = self.plausibility_checker.check_with_llm(scenarios_for_llm)

            # === NEW: Apply corroboration filter ===
            # Combine all deterministic issues for corroboration check
            all_deterministic_issues = {**all_issues_1a, **all_issues_1b, **all_issues_1c}
            for k, v in all_issues_1d.items():
                if k in all_deterministic_issues:
                    all_deterministic_issues[k].extend(v)
                else:
                    all_deterministic_issues[k] = v

            # Get corroboration summary before filtering
            self.llm_corroboration_summary = self.plausibility_checker.get_corroboration_summary(
                raw_llm_issues, all_deterministic_issues
            )

            # Filter LLM issues with corroboration
            filtered_llm_issues = self.plausibility_checker.filter_llm_issues_with_corroboration(
                raw_llm_issues, all_deterministic_issues
            )

            for scenario_id, issues in filtered_llm_issues.items():
                # Find the key for this scenario
                for key in self._annotations.keys():
                    if key[1] == scenario_id:
                        all_issues_1d[key].extend(issues)
                        break

        # Aggregate scores
        logger.info("\n[Aggregation] Computing quality scores...")
        self._aggregate_all_scores(
            all_issues_1a,
            all_issues_1b,
            all_issues_1c,
            all_issues_1d,
        )

        # === NEW: Generate per-annotator aggregate reports ===
        logger.info("\n[Annotator Analysis] Generating per-annotator reports...")
        self.annotator_reports = self.sampler.generate_annotator_reports(
            self.file_reports, self.quality_reports
        )
        logger.info(f"  Generated reports for {len(self.annotator_reports)} annotators")

        # Generate sampling plan
        logger.info("\n[Sampling] Generating human review plan...")
        sampling_plan = self.sampler.generate_plan(
            self.quality_reports,
            self.file_reports,
            self.fleiss_kappa_by_subtype,
        )

        # Generate text report
        report_text = self.sampler.generate_text_report(sampling_plan)
        print("\n" + report_text)

        # Save outputs
        if self.config.output.generate.sampling_plan:
            self._save_outputs(sampling_plan, report_text)

        return sampling_plan

    def _aggregate_all_scores(
        self,
        issues_1a: Dict[tuple[str, int], List[QualityIssue]],
        issues_1b: Dict[tuple[str, int], List[QualityIssue]],
        issues_1c: Dict[tuple[str, int], List[QualityIssue]],
        issues_1d: Dict[tuple[str, int], List[QualityIssue]],
    ) -> None:
        """Aggregate all issues into quality reports."""
        n_scenarios = self.config.schema_config.scenarios_per_subtype

        for subtype in PragmaticSubtype:
            subtype_name = subtype.value

            for scenario_id in range(1, n_scenarios + 1):
                key = (subtype_name, scenario_id)

                scenario = self._scenarios.get(key)
                annotations = self._annotations.get(key, [])

                # Build display text
                scenario_text = ""
                annotations_summary: Dict[str, Any] = {}

                if scenario:
                    scenario_text = scenario.to_display_text()
                    annotations_summary = {
                        "emotions": [a.plutchik_emotion for a in annotations],
                        "valence": [a.valence for a in annotations],
                        "arousal": [a.arousal for a in annotations],
                        "dominance": [a.dominance for a in annotations],
                        "confidence": [a.confidence for a in annotations],
                        "annotators": [a.annotator_name for a in annotations],
                    }

                # Aggregate
                report = self.score_aggregator.aggregate_scenario(
                    scenario_id=scenario_id,
                    task_id=scenario.task_id if scenario else 0,
                    subtype=subtype_name,
                    file_paths=list({a.file_path for a in annotations}),
                    issues_1a=issues_1a.get(key, []),
                    issues_1b=issues_1b.get(key, []),
                    issues_1c=issues_1c.get(key, []),
                    issues_1d=issues_1d.get(key, []),
                    scenario_text=scenario_text,
                    annotations_summary=annotations_summary,
                )

                self.quality_reports.append(report)

    def _save_outputs(self, plan: SamplingPlan, report_text: str) -> None:
        """Save all outputs to disk."""
        output_dir = self.paths.output_dir
        output_config = self.config.output
        indent = output_config.json_indent

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Sampling plan
        if output_config.generate.sampling_plan:
            path = output_dir / "sampling_plan.json"
            with open(path, "w") as f:
                json.dump(plan.to_dict(), f, indent=indent)
            logger.info(f"  Saved: {path}")

        # All quality reports
        if output_config.generate.all_reports:
            path = output_dir / "all_quality_reports.json"
            with open(path, "w") as f:
                json.dump([r.to_dict() for r in self.quality_reports], f, indent=indent)
            logger.info(f"  Saved: {path}")

        # File reports
        if output_config.generate.file_reports:
            path = output_dir / "file_reports.json"
            with open(path, "w") as f:
                json.dump([r.to_dict() for r in self.file_reports], f, indent=indent)
            logger.info(f"  Saved: {path}")

        # Issues summary
        if output_config.generate.issues_summary:
            summary = self._build_issues_summary()
            path = output_dir / "issues_summary.json"
            with open(path, "w") as f:
                json.dump(summary, f, indent=indent)
            logger.info(f"  Saved: {path}")

        # Review queue
        if output_config.generate.review_queue:
            queue = self._build_review_queue(output_config.review_queue_max_items)
            path = output_dir / "review_queue.json"
            with open(path, "w") as f:
                json.dump(queue, f, indent=indent)
            logger.info(f"  Saved: {path}")

        # Text report
        if output_config.generate.text_report:
            path = output_dir / "review_report.txt"
            with open(path, "w") as f:
                f.write(report_text)
            logger.info(f"  Saved: {path}")

        # === NEW OUTPUTS ===

        # Per-annotator aggregate reports
        if getattr(output_config.generate, "annotator_reports", True):
            path = output_dir / "annotator_reports.json"
            with open(path, "w") as f:
                json.dump([r.to_dict() for r in self.annotator_reports], f, indent=indent)
            logger.info(f"  Saved: {path}")

        # Confusion matrix
        if getattr(output_config.generate, "confusion_matrix", True):
            path = output_dir / "confusion_matrix.json"
            with open(path, "w") as f:
                json.dump(self._build_confusion_matrix_output(), f, indent=indent)
            logger.info(f"  Saved: {path}")

        # Agreement metrics (with CIs and weighted kappa)
        if getattr(output_config.generate, "agreement_metrics", True):
            path = output_dir / "agreement_metrics.json"
            with open(path, "w") as f:
                json.dump(self._build_agreement_metrics(), f, indent=indent)
            logger.info(f"  Saved: {path}")

        # Run metadata
        if getattr(output_config.generate, "run_metadata", True) and self.run_metadata:
            path = output_dir / "run_metadata.json"
            with open(path, "w") as f:
                json.dump(self.run_metadata.to_dict(), f, indent=indent)
            logger.info(f"  Saved: {path}")

        # LLM corroboration summary (if LLM was used)
        if self.llm_corroboration_summary:
            path = output_dir / "llm_corroboration.json"
            with open(path, "w") as f:
                json.dump(self.llm_corroboration_summary, f, indent=indent)
            logger.info(f"  Saved: {path}")

        # === REVIEW REPORTS (to reports/ directory) ===
        self._save_review_reports(plan)

        logger.info(f"\nOutputs saved to: {output_dir}")

    def _save_review_reports(self, plan: SamplingPlan) -> None:
        """Save human review reports to reports/ directory."""
        # Ensure reports directory exists
        reports_dir = self.paths.reports_dir
        reports_dir.mkdir(parents=True, exist_ok=True)

        indent = self.config.output.json_indent

        # Build annotations by scenario for review report
        annotations_by_scenario: Dict[int, List[AnnotationData]] = {}
        for (subtype_name, scenario_id), annotations in self._annotations.items():
            if scenario_id not in annotations_by_scenario:
                annotations_by_scenario[scenario_id] = []
            annotations_by_scenario[scenario_id].extend(annotations)

        # Compute differentiation scores if not already done
        if not self.differentiation_scores:
            for report in self.quality_reports:
                annotations = annotations_by_scenario.get(report.scenario_id, [])
                diff_score = self.differentiation_calculator.compute_differentiated_score(
                    scenario_id=report.scenario_id,
                    subtype=report.subtype,
                    base_quality_score=report.quality_score,
                    annotations=annotations,
                )
                self.differentiation_scores[report.scenario_id] = diff_score

        # Generate QA review report
        config_hash = self.run_metadata.config_hash if self.run_metadata else ""
        generator = ReviewReportGenerator(self.config)
        review_report = generator.generate(
            sampling_plan=plan,
            annotations_by_scenario=annotations_by_scenario,
            differentiation_scores=self.differentiation_scores,
            config_hash=config_hash,
        )

        # Save report_qa_review.json
        path = reports_dir / "report_qa_review.json"
        generator.save(review_report, path)
        logger.info(f"  Saved: {path}")

        # Save differentiation scores
        diff_path = reports_dir / "differentiation_scores.json"
        with open(diff_path, "w") as f:
            json.dump(
                {str(k): v.to_dict() for k, v in self.differentiation_scores.items()},
                f,
                indent=indent,
            )
        logger.info(f"  Saved: {diff_path}")

        # Generate markdown summary report (records only)
        self._save_markdown_report(plan, reports_dir)

        # Generate comprehensive QA report
        comprehensive_path = reports_dir / "report_qa_comprehensive.md"
        generate_comprehensive_report(self, comprehensive_path)
        logger.info(f"  Saved: {comprehensive_path}")

    def _save_markdown_report(self, plan: SamplingPlan, reports_dir: Path) -> None:
        """Generate human-readable markdown report."""
        lines = [
            "# CEI Quality Pipeline - Human Review Report",
            "",
            f"**Generated:** {datetime.now().isoformat()}",
            f"**Config Hash:** {self.run_metadata.config_hash if self.run_metadata else 'N/A'}",
            "",
            "## Summary",
            "",
            f"- **Total Scenarios:** {plan.total_scenarios}",
            f"- **Scenarios Flagged:** {plan.scenarios_flagged} ({100*plan.scenarios_flagged/max(1,plan.total_scenarios):.1f}%)",
            f"- **Selected for Review:** {plan.sample_size}",
            f"  - Mandatory: {len(plan.mandatory_reviews)}",
            f"  - Stratified Sample: {len(plan.stratified_sample)}",
            "",
            "## Inter-Annotator Agreement",
            "",
            f"**Overall Fleiss' κ:** {plan.overall_fleiss_kappa:.3f}",
            "",
            "| Subtype | Fleiss' κ | Interpretation |",
            "|---------|-----------|----------------|",
        ]

        for subtype, kappa in sorted(plan.fleiss_kappa_by_subtype.items()):
            interp = self.agreement_validator.interpret_kappa(kappa)
            lines.append(f"| {subtype} | {kappa:.3f} | {interp} |")

        lines.extend([
            "",
            "## Priority Review Items",
            "",
            "Items are ordered by priority (highest first).",
            "",
        ])

        # All mandatory items (no limit)
        for i, item in enumerate(plan.mandatory_reviews, 1):
            diff_score = self.differentiation_scores.get(item.scenario_id)
            diff_str = f" | Diff: {diff_score.differentiated_score:.3f}" if diff_score else ""

            lines.extend([
                f"### {i}. [{item.subtype}] Scenario {item.scenario_id}",
                "",
                f"- **Quality Score:** {item.quality_score:.3f}{diff_str}",
                f"- **Priority:** {item.review_priority}",
                f"- **Review Reasons:**",
            ])
            for reason in item.review_reasons:
                lines.append(f"  - {reason}")

            if diff_score and diff_score.agreement_metrics:
                am = diff_score.agreement_metrics
                lines.extend([
                    f"- **Emotion Dispersion:** {am.emotion_dispersion:.3f}",
                    f"- **Contains Opposites:** {am.contains_opposites}",
                    f"- **Has Majority:** {am.has_majority} ({am.majority_emotion or 'N/A'})",
                ])

            # Add annotator details for manual review
            annotations = self._annotations.get((item.subtype, item.scenario_id), [])
            if annotations:
                lines.extend([
                    "",
                    "#### Annotator Labels (for manual correction)",
                    "",
                    "| Annotator | Emotion | V | A | D | File |",
                    "|-----------|---------|---|---|---|------|",
                ])
                for ann in annotations:
                    # Extract just the filename from path
                    filename = Path(ann.file_path).name if ann.file_path else "N/A"
                    # Truncate filename for display
                    short_file = filename[:40] + "..." if len(filename) > 40 else filename
                    lines.append(
                        f"| {ann.annotator_name} | {ann.plutchik_emotion} | "
                        f"{ann.valence[:10] if ann.valence else 'N/A'} | "
                        f"{ann.arousal[:10] if ann.arousal else 'N/A'} | "
                        f"{ann.dominance[:10] if ann.dominance else 'N/A'} | "
                        f"`{short_file}` |"
                    )

            lines.append("")

        # Save
        path = reports_dir / "report_qa_records.md"
        with open(path, "w") as f:
            f.write("\n".join(lines))
        logger.info(f"  Saved: {path}")

    def _build_issues_summary(self) -> Dict[str, Any]:
        """Build aggregate issues summary."""
        summary: Dict[str, Any] = {
            "by_stage": {
                "1A_schema": 0,
                "1B_consistency": 0,
                "1C_agreement": 0,
                "1D_plausibility": 0,
            },
            "by_severity": {
                "critical": 0,
                "major": 0,
                "minor": 0,
                "info": 0,
            },
            "by_flag": {},
        }

        for report in self.quality_reports:
            for issue in report.issues:
                # By stage
                stage = issue.flag.stage
                stage_key = f"{stage}_" + {
                    "1A": "schema",
                    "1B": "consistency",
                    "1C": "agreement",
                    "1D": "plausibility",
                }.get(stage, "unknown")
                summary["by_stage"][stage_key] = summary["by_stage"].get(stage_key, 0) + 1

                # By severity
                summary["by_severity"][issue.severity] += 1

                # By flag
                flag_name = issue.flag.value
                summary["by_flag"][flag_name] = summary["by_flag"].get(flag_name, 0) + 1

        return summary

    def _build_review_queue(self, max_items: int) -> List[Dict[str, Any]]:
        """Build prioritized review queue."""
        # Sort by priority (desc) then quality score (asc)
        review_items = sorted(
            [r for r in self.quality_reports if r.needs_review],
            key=lambda x: (-x.review_priority, x.quality_score),
        )

        return [
            {
                "rank": i + 1,
                "scenario_id": r.scenario_id,
                "subtype": r.subtype,
                "quality_score": round(r.quality_score, 3),
                "priority": r.review_priority,
                "reasons": r.review_reasons,
                "scenario_text": r.scenario_text,
                "annotations": r.annotations_summary,
            }
            for i, r in enumerate(review_items[:max_items])
        ]

    def _generate_run_metadata(self, total_files: int, total_records: int) -> RunMetadata:
        """Generate metadata about this pipeline run for reproducibility."""
        # Generate config hash
        config_str = json.dumps(self.config.model_dump(), sort_keys=True, default=str)
        config_hash = hashlib.sha256(config_str.encode()).hexdigest()[:16]

        # Get git info if available
        git_commit = None
        git_branch = None
        try:
            git_commit = (
                subprocess.check_output(
                    ["git", "rev-parse", "HEAD"],
                    stderr=subprocess.DEVNULL,
                    cwd=str(self.paths.output_dir.parent),
                )
                .decode()
                .strip()[:8]
            )
            git_branch = (
                subprocess.check_output(
                    ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                    stderr=subprocess.DEVNULL,
                    cwd=str(self.paths.output_dir.parent),
                )
                .decode()
                .strip()
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass

        # Get weight profile
        weight_profile = "balanced"
        try:
            wp = getattr(self.config.quality, "weight_profile", None)
            if wp is not None:
                if hasattr(wp, "value"):
                    weight_profile = wp.value
                else:
                    weight_profile = str(wp)
        except Exception:
            pass

        return RunMetadata(
            run_id=str(uuid.uuid4())[:8],
            timestamp=datetime.now().isoformat(),
            config_hash=config_hash,
            git_commit=git_commit,
            git_branch=git_branch,
            python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            package_version="1.0.0",  # TODO: Get from package
            data_dir=str(self.paths.data_dir),
            total_files=total_files,
            total_records=total_records,
            llm_enabled=self.config.llm.enabled,
            llm_provider=self.config.llm.provider if self.config.llm.enabled else None,
            random_seed=self.config.sampling.random_seed,
            weight_profile=str(weight_profile),
        )

    def _build_agreement_metrics(self) -> Dict[str, Any]:
        """Build comprehensive agreement metrics output."""
        return {
            "fleiss_kappa": {
                "by_subtype": self.fleiss_kappa_by_subtype,
                "overall": sum(self.fleiss_kappa_by_subtype.values())
                / max(1, len(self.fleiss_kappa_by_subtype)),
            },
            "fleiss_kappa_with_ci": self.kappa_with_ci_by_subtype,
            "weighted_kappa_vad": self.weighted_kappa_by_subtype,
            "interpretation": {
                subtype: InterAnnotatorValidator.interpret_kappa(kappa)
                for subtype, kappa in self.fleiss_kappa_by_subtype.items()
            },
        }

    def _build_confusion_matrix_output(self) -> Dict[str, Any]:
        """Build confusion matrix output for all subtypes."""
        return {
            "by_subtype": self.confusion_matrices_by_subtype,
            "overall": self._merge_confusion_matrices(),
        }

    def _merge_confusion_matrices(self) -> Dict[str, Any]:
        """Merge confusion matrices across all subtypes."""
        if not self.confusion_matrices_by_subtype:
            return {}

        # Get emotion list from first matrix
        first = next(iter(self.confusion_matrices_by_subtype.values()))
        emotions = first.get("emotions", [])

        # Initialize merged matrix
        merged: Dict[str, Dict[str, int]] = {e: {e2: 0 for e2 in emotions} for e in emotions}

        # Sum across subtypes
        for subtype_matrix in self.confusion_matrices_by_subtype.values():
            matrix = subtype_matrix.get("matrix", {})
            for e1, row in matrix.items():
                for e2, count in row.items():
                    merged[e1][e2] += count

        # Compute overall statistics
        total = sum(sum(row.values()) for row in merged.values())
        diagonal = sum(merged[e][e] for e in emotions)

        return {
            "matrix": merged,
            "emotions": emotions,
            "total_pairs": total // 2,
            "agreement_count": diagonal,
            "agreement_rate": round(diagonal / max(1, total), 4),
        }
