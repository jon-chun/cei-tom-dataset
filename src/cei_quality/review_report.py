"""
Human Review Report Generator for CEI Quality Pipeline.

This module generates machine-readable and human-friendly review reports
that identify problematic scenarios for human QA review.

Output: reports/report_qa_review.json
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from cei_quality.config import CEIConfig
    from cei_quality.models import (
        AnnotationData,
        RecordQualityReport,
        SamplingPlan,
    )
    from cei_quality.differentiation import DifferentiationScore, ScenarioAgreementMetrics


@dataclass
class ReviewRecord:
    """
    A single record for human QA review.

    Contains all necessary identifiers and metrics to uniquely identify
    and prioritize a scenario for review.
    """

    # Priority (1 = highest priority, lower score = higher priority)
    priority: int

    # Unique identifiers
    scenario_type: str  # e.g., "sarcasm-irony", "mixed-signals"
    annotator_id: str  # annotations.completed_by
    data_id: int  # data.id within the file

    # Metrics
    metrics: Dict[str, Any] = field(default_factory=dict)

    # Review status
    reviewed: bool = False
    prior_values: List[Dict[str, Any]] = field(default_factory=list)

    # Additional context
    file_path: str = ""
    task_id: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "priority": self.priority,
            "scenario_type": self.scenario_type,
            "annotator_id": self.annotator_id,
            "data_id": self.data_id,
            "metrics": self.metrics,
            "reviewed": self.reviewed,
            "prior_values": self.prior_values,
            "file_path": self.file_path,
            "task_id": self.task_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReviewRecord":
        """Create ReviewRecord from dictionary."""
        return cls(
            priority=data["priority"],
            scenario_type=data["scenario_type"],
            annotator_id=data["annotator_id"],
            data_id=data["data_id"],
            metrics=data.get("metrics", {}),
            reviewed=data.get("reviewed", False),
            prior_values=data.get("prior_values", []),
            file_path=data.get("file_path", ""),
            task_id=data.get("task_id", 0),
        )

    @property
    def unique_key(self) -> str:
        """Generate unique key for this record."""
        return f"{self.scenario_type}|{self.annotator_id}|{self.data_id}"


@dataclass
class ScenarioReviewGroup:
    """
    A group of annotator records for a single scenario.

    Provides scenario-level view for meta-review with all annotator
    labels visible together.
    """

    # Scenario identifiers
    scenario_id: int
    scenario_type: str  # e.g., "sarcasm-irony"
    priority: int
    task_id: int = 0

    # Review reasons (from quality flags)
    review_reasons: List[str] = field(default_factory=list)
    quality_score: float = 0.0

    # All annotator records for this scenario
    annotator_records: List[ReviewRecord] = field(default_factory=list)

    # Status
    reviewed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "scenario_id": self.scenario_id,
            "scenario_type": self.scenario_type,
            "priority": self.priority,
            "task_id": self.task_id,
            "review_reasons": self.review_reasons,
            "quality_score": round(self.quality_score, 4),
            "reviewed": self.reviewed,
            "annotator_count": len(self.annotator_records),
            "annotators": [
                {
                    "annotator_id": r.annotator_id,
                    "file_path": r.file_path,
                    "data_id": r.data_id,
                    "metrics": r.metrics,
                    "reviewed": r.reviewed,
                    "prior_values": r.prior_values,
                }
                for r in self.annotator_records
            ],
        }


@dataclass
class ReviewReport:
    """
    Complete human review report.

    Contains all records flagged for review, sorted by priority.
    """

    # Metadata
    generated_at: str
    config_hash: str
    total_scenarios: int
    flagged_count: int

    # Review records (flat list, one per annotator per scenario)
    records: List[ReviewRecord] = field(default_factory=list)

    # Scenario-grouped records (one per flagged scenario, contains all annotators)
    scenario_groups: List[ScenarioReviewGroup] = field(default_factory=list)

    # Summary statistics
    by_scenario_type: Dict[str, int] = field(default_factory=dict)
    by_priority_level: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "metadata": {
                "generated_at": self.generated_at,
                "config_hash": self.config_hash,
                "total_scenarios": self.total_scenarios,
                "flagged_count": self.flagged_count,
                "flagged_scenario_count": len(self.scenario_groups),
            },
            "summary": {
                "by_scenario_type": self.by_scenario_type,
                "by_priority_level": self.by_priority_level,
            },
            # Scenario-grouped view (recommended for meta-review)
            "scenarios": [g.to_dict() for g in self.scenario_groups],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReviewReport":
        """Create ReviewReport from dictionary."""
        metadata = data.get("metadata", {})
        summary = data.get("summary", {})

        report = cls(
            generated_at=metadata.get("generated_at", ""),
            config_hash=metadata.get("config_hash", ""),
            total_scenarios=metadata.get("total_scenarios", 0),
            flagged_count=metadata.get("flagged_count", 0),
            by_scenario_type=summary.get("by_scenario_type", {}),
            by_priority_level=summary.get("by_priority_level", {}),
        )

        # Load scenario groups (new format)
        for scenario_data in data.get("scenarios", []):
            group = ScenarioReviewGroup(
                scenario_id=scenario_data["scenario_id"],
                scenario_type=scenario_data["scenario_type"],
                priority=scenario_data["priority"],
                task_id=scenario_data.get("task_id", 0),
                review_reasons=scenario_data.get("review_reasons", []),
                quality_score=scenario_data.get("quality_score", 0.0),
                reviewed=scenario_data.get("reviewed", False),
            )
            for ann_data in scenario_data.get("annotators", []):
                record = ReviewRecord(
                    priority=group.priority,
                    scenario_type=group.scenario_type,
                    annotator_id=ann_data["annotator_id"],
                    data_id=ann_data["data_id"],
                    metrics=ann_data.get("metrics", {}),
                    reviewed=ann_data.get("reviewed", False),
                    prior_values=ann_data.get("prior_values", []),
                    file_path=ann_data.get("file_path", ""),
                    task_id=group.task_id,
                )
                group.annotator_records.append(record)
                report.records.append(record)
            report.scenario_groups.append(group)

        return report

    def get_next_unreviewed(self) -> Optional[ReviewRecord]:
        """Get next unreviewed record in priority order."""
        for record in self.records:
            if not record.reviewed:
                return record
        return None

    def get_next_unreviewed_scenario(self) -> Optional[ScenarioReviewGroup]:
        """Get next unreviewed scenario group in priority order."""
        for group in self.scenario_groups:
            if not group.reviewed:
                return group
        return None

    def mark_scenario_reviewed(
        self,
        scenario_id: int,
        scenario_type: str,
        changes: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    ) -> bool:
        """
        Mark an entire scenario as reviewed and record any changes.

        Args:
            scenario_id: Scenario ID (data.id)
            scenario_type: Scenario type identifier
            changes: Dict mapping annotator_id to list of changes

        Returns:
            True if scenario was found and updated
        """
        for group in self.scenario_groups:
            if group.scenario_id == scenario_id and group.scenario_type == scenario_type:
                group.reviewed = True
                # Mark all annotator records as reviewed
                for record in group.annotator_records:
                    record.reviewed = True
                    if changes and record.annotator_id in changes:
                        record.prior_values.extend(changes[record.annotator_id])
                return True
        return False

    def mark_reviewed(
        self,
        scenario_type: str,
        annotator_id: str,
        data_id: int,
        changes: Optional[List[Dict[str, Any]]] = None,
    ) -> bool:
        """
        Mark a record as reviewed and record any changes.

        Args:
            scenario_type: Scenario type identifier
            annotator_id: Annotator identifier
            data_id: Data ID
            changes: List of changes made, each as {"field": "old_value"}

        Returns:
            True if record was found and updated
        """
        key = f"{scenario_type}|{annotator_id}|{data_id}"
        for record in self.records:
            if record.unique_key == key:
                record.reviewed = True
                if changes:
                    record.prior_values.extend(changes)
                return True
        return False


class ReviewReportGenerator:
    """
    Generator for human QA review reports.

    Creates prioritized review lists with unique identifiers
    for each flagged scenario.
    """

    def __init__(self, config: "CEIConfig") -> None:
        self.config = config

    def generate(
        self,
        sampling_plan: "SamplingPlan",
        annotations_by_scenario: Dict[int, List["AnnotationData"]],
        differentiation_scores: Optional[Dict[int, "DifferentiationScore"]] = None,
        config_hash: str = "",
    ) -> ReviewReport:
        """
        Generate a review report from a sampling plan.

        Args:
            sampling_plan: The sampling plan with mandatory and stratified reviews
            annotations_by_scenario: Mapping of scenario_id to annotations
            differentiation_scores: Optional enhanced scores
            config_hash: Hash of the configuration used

        Returns:
            ReviewReport ready for human review
        """
        report = ReviewReport(
            generated_at=datetime.now().isoformat(),
            config_hash=config_hash,
            total_scenarios=sampling_plan.total_scenarios,
            flagged_count=len(sampling_plan.total_review_items),
        )

        # Combine all review items
        all_items = sampling_plan.mandatory_reviews + sampling_plan.stratified_sample

        # Create review records and scenario groups
        priority_counter = 0
        for item in all_items:
            priority_counter += 1

            # Get annotations for this scenario
            annotations = annotations_by_scenario.get(item.scenario_id, [])

            # Create scenario group (one per flagged scenario)
            scenario_group = ScenarioReviewGroup(
                scenario_id=item.scenario_id,
                scenario_type=item.subtype,
                priority=priority_counter,
                task_id=item.task_id,
                review_reasons=item.review_reasons,
                quality_score=item.quality_score,
            )

            # Each annotation from each annotator gets a record within the group
            for ann in annotations:
                metrics = self._build_metrics(item, ann, differentiation_scores)

                record = ReviewRecord(
                    priority=priority_counter,
                    scenario_type=item.subtype,
                    annotator_id=ann.annotator_name,
                    data_id=item.scenario_id,
                    metrics=metrics,
                    file_path=ann.file_path,
                    task_id=item.task_id,
                )
                report.records.append(record)
                scenario_group.annotator_records.append(record)

            report.scenario_groups.append(scenario_group)

        # Sort by priority (lower number = higher priority)
        report.records.sort(key=lambda r: r.priority)
        report.scenario_groups.sort(key=lambda g: g.priority)

        # Compute summary statistics (per scenario, not per annotator)
        report.by_scenario_type = {}
        report.by_priority_level = {"critical": 0, "high": 0, "medium": 0, "low": 0}

        for group in report.scenario_groups:
            # By scenario type
            st = group.scenario_type
            report.by_scenario_type[st] = report.by_scenario_type.get(st, 0) + 1

            # By priority level
            if group.priority <= 10:
                report.by_priority_level["critical"] += 1
            elif group.priority <= 30:
                report.by_priority_level["high"] += 1
            elif group.priority <= 60:
                report.by_priority_level["medium"] += 1
            else:
                report.by_priority_level["low"] += 1

        return report

    def _build_metrics(
        self,
        item: "RecordQualityReport",
        annotation: "AnnotationData",
        differentiation_scores: Optional[Dict[int, "DifferentiationScore"]] = None,
    ) -> Dict[str, Any]:
        """Build metrics dictionary for a review record."""
        metrics = {
            "quality_score": round(item.quality_score, 4),
            "schema_score": round(item.schema_score, 4),
            "consistency_score": round(item.consistency_score, 4),
            "agreement_score": round(item.agreement_score, 4),
            "plausibility_score": round(item.plausibility_score, 4),
            "review_priority": item.review_priority,
            "review_reasons": item.review_reasons,
            "issue_count": item.issue_count,
            "critical_count": item.critical_count,
            "major_count": item.major_count,
            "lead_time": annotation.lead_time,
        }

        # Add differentiation metrics if available
        if differentiation_scores and item.scenario_id in differentiation_scores:
            diff_score = differentiation_scores[item.scenario_id]
            metrics["differentiated_score"] = round(diff_score.differentiated_score, 4)
            metrics["dispersion_factor"] = round(diff_score.dispersion_factor, 4)
            metrics["timing_factor"] = round(diff_score.timing_factor, 4)

            if diff_score.agreement_metrics:
                am = diff_score.agreement_metrics
                metrics["emotion_dispersion"] = round(am.emotion_dispersion, 4)
                metrics["contains_opposites"] = am.contains_opposites
                metrics["has_majority"] = am.has_majority
                metrics["majority_emotion"] = am.majority_emotion

        return metrics

    def save(self, report: ReviewReport, output_path: Path) -> None:
        """
        Save review report to JSON file.

        Args:
            report: ReviewReport to save
            output_path: Path to output file
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(report.to_dict(), f, indent=2)

    def load(self, input_path: Path) -> ReviewReport:
        """
        Load review report from JSON file.

        Args:
            input_path: Path to input file

        Returns:
            Loaded ReviewReport
        """
        with open(input_path, "r") as f:
            data = json.load(f)

        return ReviewReport.from_dict(data)


def generate_qa_review_report(
    config: "CEIConfig",
    sampling_plan: "SamplingPlan",
    annotations_by_scenario: Dict[int, List["AnnotationData"]],
    differentiation_scores: Optional[Dict[int, "DifferentiationScore"]] = None,
    config_hash: str = "",
) -> Path:
    """
    Convenience function to generate and save QA review report.

    Args:
        config: CEI configuration
        sampling_plan: Sampling plan with review items
        annotations_by_scenario: Annotations indexed by scenario ID
        differentiation_scores: Optional enhanced scores
        config_hash: Configuration hash

    Returns:
        Path to the generated report file
    """
    generator = ReviewReportGenerator(config)

    report = generator.generate(
        sampling_plan=sampling_plan,
        annotations_by_scenario=annotations_by_scenario,
        differentiation_scores=differentiation_scores,
        config_hash=config_hash,
    )

    # Get reports directory from config
    paths = config.get_resolved_paths()
    output_path = paths.reports_dir / "report_qa_review.json"

    generator.save(report, output_path)

    return output_path
