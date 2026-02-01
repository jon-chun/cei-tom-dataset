"""
Data models for CEI Quality Pipeline.

This module defines all data structures used throughout the pipeline:
- Quality flags and issues
- Scenario and annotation data
- Quality reports
- Sampling plans
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import Counter

import numpy as np


# =============================================================================
# Enumerations
# =============================================================================


class PragmaticSubtype(Enum):
    """The 5 pragmatic subtypes in CEI."""

    SARCASM_IRONY = "sarcasm-irony"
    MIXED_SIGNALS = "mixed-signals"
    STRATEGIC_POLITENESS = "strategic-politeness"
    PASSIVE_AGGRESSION = "passive-aggression"
    DEFLECTION_MISDIRECTION = "deflection-misdirection"

    @classmethod
    def from_string(cls, s: str) -> Optional["PragmaticSubtype"]:
        """Convert string to subtype enum, handling variations."""
        normalized = s.lower().replace("_", "-").replace(" ", "-")

        # Direct match
        for subtype in cls:
            if subtype.value == normalized:
                return subtype

        # Partial match
        for subtype in cls:
            if subtype.value.replace("-", "") in normalized.replace("-", ""):
                return subtype

        return None

    @classmethod
    def from_filename(cls, filename: str) -> Optional["PragmaticSubtype"]:
        """Extract subtype from filename."""
        filename_lower = filename.lower()
        for subtype in cls:
            # Check if subtype identifier is in filename
            subtype_variants = [
                subtype.value,
                subtype.value.replace("-", ""),
                subtype.value.replace("-", "_"),
            ]
            for variant in subtype_variants:
                if variant in filename_lower:
                    return subtype
        return None


class QualityFlag(Enum):
    """
    Taxonomy of quality issues by pipeline stage.

    Naming convention: {STAGE}_{ISSUE_TYPE}
    - 1A: Schema validation
    - 1B: Within-file consistency
    - 1C: Inter-annotator agreement
    - 1D: LLM plausibility
    """

    # Stage 1A: Schema/Missing Data
    MISSING_REQUIRED_FIELD = "1A_missing_required_field"
    MALFORMED_JSON = "1A_malformed_json"
    EMPTY_VALUE = "1A_empty_value"
    INVALID_DATA_TYPE = "1A_invalid_data_type"
    NO_ANNOTATIONS = "1A_no_annotations"
    INCOMPLETE_ANNOTATION = "1A_incomplete_annotation"

    # Stage 1B: Within-File Consistency
    DUPLICATE_SCENARIO_ID = "1B_duplicate_scenario_id"
    INVALID_LABEL_VALUE = "1B_invalid_label_value"
    SUSPICIOUS_LEAD_TIME = "1B_suspicious_lead_time"
    TIMESTAMP_ANOMALY = "1B_timestamp_anomaly"
    UNEXPECTED_SCENARIO_COUNT = "1B_unexpected_scenario_count"
    MISSING_SCENARIO_ID = "1B_missing_scenario_id"
    STRAIGHT_LINING = "1B_straight_lining"
    DWELL_TIME_OUTLIER = "1B_dwell_time_outlier"
    SELF_CONTRADICTION = "1B_self_contradiction"
    ANNOTATOR_DWELL_OUTLIER = "1B_annotator_dwell_outlier"

    # Stage 1C: Inter-Annotator Agreement
    MISSING_CROSS_ANNOTATION = "1C_missing_cross_annotation"
    HIGH_EMOTION_DISAGREEMENT = "1C_high_emotion_disagreement"
    HIGH_VAD_DISAGREEMENT = "1C_high_vad_disagreement"
    OUTLIER_ANNOTATOR = "1C_outlier_annotator"
    NO_MAJORITY_EMOTION = "1C_no_majority_emotion"

    # Stage 1D: LLM Plausibility
    IMPLAUSIBLE_LISTENER_EMOTION = "1D_implausible_listener_emotion"
    VAD_EMOTION_MISMATCH = "1D_vad_emotion_mismatch"
    INCOHERENT_SCENARIO = "1D_incoherent_scenario"
    SPEAKER_LISTENER_CONFUSION = "1D_speaker_listener_confusion"

    @property
    def stage(self) -> str:
        """Get the stage this flag belongs to."""
        return self.value.split("_")[0]

    @property
    def stage_name(self) -> str:
        """Get human-readable stage name."""
        stage_names = {
            "1A": "Schema Validation",
            "1B": "Within-File Consistency",
            "1C": "Inter-Annotator Agreement",
            "1D": "LLM Plausibility",
        }
        return stage_names.get(self.stage, "Unknown")


class Severity(Enum):
    """Issue severity levels."""

    CRITICAL = "critical"
    MAJOR = "major"
    MINOR = "minor"
    INFO = "info"

    @property
    def priority(self) -> int:
        """Get numeric priority (higher = more severe)."""
        return {"critical": 4, "major": 3, "minor": 2, "info": 1}[self.value]


# =============================================================================
# Data Classes: Issues
# =============================================================================


@dataclass
class QualityIssue:
    """
    A single quality issue detected during validation.

    Attributes:
        flag: The type of quality issue
        severity: How severe the issue is
        message: Human-readable description
        details: Additional context and data
    """

    flag: QualityFlag
    severity: str
    message: str
    details: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate severity value."""
        valid_severities = {"critical", "major", "minor", "info"}
        if self.severity not in valid_severities:
            raise ValueError(
                f"Invalid severity: {self.severity}. Must be one of {valid_severities}"
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "flag": self.flag.value,
            "flag_stage": self.flag.stage,
            "severity": self.severity,
            "message": self.message,
            "details": self.details,
        }

    @property
    def is_critical(self) -> bool:
        """Check if this is a critical issue."""
        return self.severity == "critical"

    @property
    def is_major(self) -> bool:
        """Check if this is a major issue."""
        return self.severity == "major"


# =============================================================================
# Data Classes: Scenarios and Annotations
# =============================================================================


@dataclass
class AnnotatorFile:
    """
    Metadata about a single annotator's file.

    Attributes:
        file_path: Path to the JSON file
        annotator_name: Name/ID of the annotator
        subtype: Pragmatic subtype this file covers
        project_id: Label Studio project ID
        timestamp: Extraction timestamp from filename
        records: List of parsed JSON records
    """

    file_path: Path
    annotator_name: str
    subtype: PragmaticSubtype
    project_id: int
    timestamp: str
    records: List[Dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_filename(cls, file_path: Path) -> "AnnotatorFile":
        """
        Parse filename to extract metadata.

        Expected patterns:
        - project-{N}_{annotator}-{subtype}_at-{timestamp}-{hash}.json
        - project-{N}_{annotator}-{subtype}-at-{timestamp}-{hash}.json
        """
        import re

        filename = file_path.name

        # Extract project ID
        project_match = re.search(r"project-(\d+)", filename)
        project_id = int(project_match.group(1)) if project_match else 0

        # Extract annotator name
        # Pattern: after project-N_ or project-N-, before subtype keyword
        annotator_patterns = [
            r"project-\d+[_-]([a-zA-Z]+(?:-[a-zA-Z]+)?)[_-](?:sarcasm|mixed|strategic|passive|deflection)",
            r"project-\d+[_-]([a-zA-Z-]+?)[_-](?:sarcasm|mixed|strategic|passive|deflection)",
        ]

        annotator_name = "unknown"
        for pattern in annotator_patterns:
            match = re.search(pattern, filename, re.IGNORECASE)
            if match:
                annotator_name = match.group(1).lower().replace("-", "_")
                break

        # Extract subtype
        subtype = PragmaticSubtype.from_filename(filename)
        if subtype is None:
            subtype = PragmaticSubtype.SARCASM_IRONY  # Default fallback

        # Extract timestamp
        timestamp_match = re.search(r"at-(\d{4}-\d{2}-\d{2}-\d{2}-\d{2})", filename)
        timestamp = timestamp_match.group(1) if timestamp_match else ""

        return cls(
            file_path=file_path,
            annotator_name=annotator_name,
            subtype=subtype,
            project_id=project_id,
            timestamp=timestamp,
        )

    def __hash__(self) -> int:
        return hash(str(self.file_path))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AnnotatorFile):
            return NotImplemented
        return self.file_path == other.file_path


@dataclass
class ScenarioData:
    """
    Data for a single scenario.

    Attributes:
        scenario_id: Unique scenario ID within subtype (1-60)
        task_id: Label Studio task ID
        situation: The situational context
        utterance: The speaker's utterance
        speaker_role: Role/relationship of the speaker
        listener_role: Role/relationship of the listener
        source_file: File this scenario was loaded from
    """

    scenario_id: int
    task_id: int
    situation: str
    utterance: str
    speaker_role: str
    listener_role: str
    source_file: str

    def __hash__(self) -> int:
        return hash((self.scenario_id, self.source_file))

    def to_display_text(self) -> str:
        """Format scenario for human review."""
        return (
            f"Situation: {self.situation}\n"
            f"Speaker ({self.speaker_role}) → Listener ({self.listener_role}):\n"
            f'"{self.utterance}"'
        )


@dataclass
class AnnotationData:
    """
    A single annotation for a scenario.

    Note: Labels describe the LISTENER's emotional response,
    not the speaker's emotion.

    Attributes:
        scenario_id: ID of the annotated scenario
        annotator_id: Label Studio user ID
        annotator_name: Human-readable annotator name
        file_path: Source file path
        plutchik_emotion: Primary Plutchik emotion
        valence: Valence rating (text)
        arousal: Arousal rating (text)
        dominance: Dominance rating (text)
        confidence: Annotator's confidence
        lead_time: Time spent on annotation (seconds)
        created_at: Timestamp of annotation
    """

    scenario_id: int
    annotator_id: str
    annotator_name: str
    file_path: str
    plutchik_emotion: str
    valence: str
    arousal: str
    dominance: str
    confidence: str
    lead_time: Optional[float] = None
    created_at: Optional[str] = None

    # Numeric mapping cache
    _vad_numeric_cache: Optional[Dict[str, float]] = field(default=None, repr=False)

    def get_vad_numeric(self, config: Any) -> Dict[str, float]:
        """
        Get numeric VAD values using config mapping.

        Args:
            config: CEIConfig instance with valid_labels

        Returns:
            Dict with keys 'v', 'a', 'd' and numeric values
        """
        if self._vad_numeric_cache is not None:
            return self._vad_numeric_cache

        self._vad_numeric_cache = {
            "v": config.get_vad_numeric("valence", self.valence),
            "a": config.get_vad_numeric("arousal", self.arousal),
            "d": config.get_vad_numeric("dominance", self.dominance),
        }
        return self._vad_numeric_cache

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "scenario_id": self.scenario_id,
            "annotator_id": self.annotator_id,
            "annotator_name": self.annotator_name,
            "plutchik_emotion": self.plutchik_emotion,
            "valence": self.valence,
            "arousal": self.arousal,
            "dominance": self.dominance,
            "confidence": self.confidence,
            "lead_time": self.lead_time,
        }


@dataclass
class ScenarioAnnotations:
    """
    All annotations for a single scenario across annotators.

    Attributes:
        scenario: The scenario data
        annotations: List of annotations (should have 3)
        subtype: Pragmatic subtype
    """

    scenario: ScenarioData
    annotations: List[AnnotationData]
    subtype: PragmaticSubtype

    @property
    def scenario_id(self) -> int:
        """Get the scenario ID."""
        return self.scenario.scenario_id

    @property
    def emotions(self) -> List[str]:
        """Get list of labeled emotions."""
        return [a.plutchik_emotion.lower().strip() for a in self.annotations if a.plutchik_emotion]

    @property
    def emotion_counts(self) -> Counter:
        """Get emotion frequency counts."""
        return Counter(self.emotions)

    @property
    def majority_emotion(self) -> Optional[str]:
        """Get majority emotion (2/3 agreement) or None."""
        if not self.annotations:
            return None
        counts = self.emotion_counts
        if not counts:
            return None
        most_common = counts.most_common(1)[0]
        if most_common[1] >= 2:  # At least 2/3 agreement
            return most_common[0]
        return None

    @property
    def emotion_agreement(self) -> float:
        """Get proportion agreeing with majority."""
        if not self.annotations:
            return 0.0
        counts = self.emotion_counts
        if not counts:
            return 0.0
        return counts.most_common(1)[0][1] / len(self.annotations)

    @property
    def is_unanimous(self) -> bool:
        """Check if all annotators agree on emotion."""
        return len(set(self.emotions)) == 1

    def get_vad_stats(self, config: Any) -> Dict[str, Dict[str, float]]:
        """
        Get VAD statistics across annotators.

        Returns:
            Dict with 'mean', 'std', 'range' for each dimension
        """
        v_vals = []
        a_vals = []
        d_vals = []

        for ann in self.annotations:
            vad = ann.get_vad_numeric(config)
            v_vals.append(vad["v"])
            a_vals.append(vad["a"])
            d_vals.append(vad["d"])

        def stats(vals: List[float]) -> Dict[str, float]:
            if not vals:
                return {"mean": 0.0, "std": 0.0, "range": 0.0}
            arr = np.array(vals)
            return {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "range": float(np.max(arr) - np.min(arr)),
            }

        return {
            "valence": stats(v_vals),
            "arousal": stats(a_vals),
            "dominance": stats(d_vals),
        }


# =============================================================================
# Data Classes: Reports
# =============================================================================


@dataclass
class RecordQualityReport:
    """
    Quality report for a single scenario.

    Aggregates issues from all pipeline stages and computes
    quality scores and review recommendations.
    """

    scenario_id: int
    task_id: int
    subtype: str
    file_paths: List[str]

    # Issues
    issues: List[QualityIssue] = field(default_factory=list)

    # Per-stage scores (0.0 to 1.0, higher = better)
    schema_score: float = 1.0
    consistency_score: float = 1.0
    agreement_score: float = 1.0
    plausibility_score: float = 1.0

    # Composite quality score
    quality_score: float = 1.0

    # Review flags
    needs_review: bool = False
    review_priority: int = 0  # 0-10, higher = more urgent
    review_reasons: List[str] = field(default_factory=list)

    # Content for review
    scenario_text: str = ""
    annotations_summary: Dict[str, Any] = field(default_factory=dict)

    @property
    def issue_count(self) -> int:
        """Total number of issues."""
        return len(self.issues)

    @property
    def critical_count(self) -> int:
        """Number of critical issues."""
        return sum(1 for i in self.issues if i.severity == "critical")

    @property
    def major_count(self) -> int:
        """Number of major issues."""
        return sum(1 for i in self.issues if i.severity == "major")

    @property
    def quality_bucket(self) -> str:
        """Get quality bucket based on score."""
        if self.quality_score >= 0.95:
            return "excellent"
        elif self.quality_score >= 0.85:
            return "good"
        elif self.quality_score >= 0.70:
            return "acceptable"
        elif self.quality_score >= 0.50:
            return "questionable"
        else:
            return "poor"

    def add_issue(self, issue: QualityIssue) -> None:
        """Add an issue to this report."""
        self.issues.append(issue)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "scenario_id": self.scenario_id,
            "task_id": self.task_id,
            "subtype": self.subtype,
            "file_paths": self.file_paths,
            "quality_score": round(self.quality_score, 4),
            "quality_bucket": self.quality_bucket,
            "schema_score": round(self.schema_score, 4),
            "consistency_score": round(self.consistency_score, 4),
            "agreement_score": round(self.agreement_score, 4),
            "plausibility_score": round(self.plausibility_score, 4),
            "needs_review": self.needs_review,
            "review_priority": self.review_priority,
            "review_reasons": self.review_reasons,
            "issue_count": self.issue_count,
            "critical_count": self.critical_count,
            "major_count": self.major_count,
            "issues": [i.to_dict() for i in self.issues],
            "scenario_text": self.scenario_text,
            "annotations_summary": self.annotations_summary,
        }


@dataclass
class FileQualityReport:
    """
    Quality report for an entire annotator file.

    Aggregates statistics across all scenarios in a file
    to identify annotator-level issues.
    """

    file_path: str
    annotator_name: str
    subtype: str

    # Counts
    record_count: int = 0
    issue_count: int = 0

    # Quality metrics
    mean_quality_score: float = 1.0
    min_quality_score: float = 1.0
    low_quality_count: int = 0  # Score < 0.7

    # Scenario tracking
    missing_scenarios: List[int] = field(default_factory=list)
    duplicate_scenarios: List[int] = field(default_factory=list)

    # Timing statistics
    mean_lead_time: float = 0.0
    min_lead_time: float = 0.0
    max_lead_time: float = 0.0
    suspicious_fast_count: int = 0
    suspicious_slow_count: int = 0

    # Overall assessment
    needs_full_review: bool = False
    review_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "file_path": self.file_path,
            "annotator_name": self.annotator_name,
            "subtype": self.subtype,
            "record_count": self.record_count,
            "issue_count": self.issue_count,
            "mean_quality_score": round(self.mean_quality_score, 4),
            "min_quality_score": round(self.min_quality_score, 4),
            "low_quality_count": self.low_quality_count,
            "missing_scenarios": self.missing_scenarios,
            "duplicate_scenarios": self.duplicate_scenarios,
            "mean_lead_time": round(self.mean_lead_time, 2),
            "suspicious_fast_count": self.suspicious_fast_count,
            "suspicious_slow_count": self.suspicious_slow_count,
            "needs_full_review": self.needs_full_review,
            "review_reason": self.review_reason,
        }


@dataclass
class AnnotatorQualityReport:
    """
    Aggregate quality report for a single annotator across all subtypes.

    Provides cross-subtype statistics to identify annotators with
    systematic issues regardless of which subtype they annotated.
    """

    annotator_name: str

    # Coverage
    subtypes_annotated: List[str] = field(default_factory=list)
    total_annotations: int = 0

    # Quality metrics
    mean_quality_score: float = 1.0
    min_quality_score: float = 1.0
    std_quality_score: float = 0.0

    # Issue aggregates
    total_issues: int = 0
    critical_issues: int = 0
    major_issues: int = 0

    # Timing statistics
    mean_lead_time: float = 0.0
    median_lead_time: float = 0.0
    lead_time_mad: float = 0.0
    dwell_outlier_count: int = 0
    dwell_outlier_rate: float = 0.0

    # Behavioral flags
    straight_lining_detected: bool = False
    self_contradictions: int = 0
    is_cohort_outlier: bool = False

    # Per-subtype breakdown
    by_subtype: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Overall assessment
    needs_full_review: bool = False
    review_reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "annotator_name": self.annotator_name,
            "subtypes_annotated": self.subtypes_annotated,
            "total_annotations": self.total_annotations,
            "mean_quality_score": round(self.mean_quality_score, 4),
            "min_quality_score": round(self.min_quality_score, 4),
            "std_quality_score": round(self.std_quality_score, 4),
            "total_issues": self.total_issues,
            "critical_issues": self.critical_issues,
            "major_issues": self.major_issues,
            "mean_lead_time": round(self.mean_lead_time, 2),
            "median_lead_time": round(self.median_lead_time, 2),
            "lead_time_mad": round(self.lead_time_mad, 2),
            "dwell_outlier_count": self.dwell_outlier_count,
            "dwell_outlier_rate": round(self.dwell_outlier_rate, 3),
            "straight_lining_detected": self.straight_lining_detected,
            "self_contradictions": self.self_contradictions,
            "is_cohort_outlier": self.is_cohort_outlier,
            "by_subtype": self.by_subtype,
            "needs_full_review": self.needs_full_review,
            "review_reasons": self.review_reasons,
        }


@dataclass
class RunMetadata:
    """
    Metadata about a pipeline run for reproducibility and auditing.
    """

    run_id: str
    timestamp: str
    config_hash: str
    git_commit: Optional[str] = None
    git_branch: Optional[str] = None
    python_version: str = ""
    package_version: str = ""

    # Input data summary
    data_dir: str = ""
    total_files: int = 0
    total_records: int = 0

    # Config highlights
    llm_enabled: bool = False
    llm_provider: Optional[str] = None
    random_seed: int = 42

    # Weight profile used
    weight_profile: str = "default"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "config_hash": self.config_hash,
            "git_commit": self.git_commit,
            "git_branch": self.git_branch,
            "python_version": self.python_version,
            "package_version": self.package_version,
            "data_dir": self.data_dir,
            "total_files": self.total_files,
            "total_records": self.total_records,
            "llm_enabled": self.llm_enabled,
            "llm_provider": self.llm_provider,
            "random_seed": self.random_seed,
            "weight_profile": self.weight_profile,
        }


@dataclass
class SamplingPlan:
    """
    Complete sampling plan for human expert review.

    Contains prioritized lists of scenarios for review and
    aggregate statistics about data quality.
    """

    # Review items
    mandatory_reviews: List[RecordQualityReport]
    stratified_sample: List[RecordQualityReport]
    files_requiring_review: List[FileQualityReport]

    # Statistics
    total_scenarios: int
    scenarios_flagged: int
    sample_size: int

    # Coverage
    coverage_by_subtype: Dict[str, int]

    # Agreement metrics
    fleiss_kappa_by_subtype: Dict[str, float]
    overall_fleiss_kappa: float = 0.0

    @property
    def mandatory_count(self) -> int:
        """Number of mandatory review items."""
        return len(self.mandatory_reviews)

    @property
    def stratified_count(self) -> int:
        """Number of stratified sample items."""
        return len(self.stratified_sample)

    @property
    def total_review_items(self) -> List[RecordQualityReport]:
        """All items selected for review."""
        return self.mandatory_reviews + self.stratified_sample

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "summary": {
                "total_scenarios": self.total_scenarios,
                "scenarios_flagged": self.scenarios_flagged,
                "flagged_rate": round(self.scenarios_flagged / max(1, self.total_scenarios), 4),
                "sample_size": self.sample_size,
                "sample_rate": round(self.sample_size / max(1, self.total_scenarios), 4),
                "mandatory_count": self.mandatory_count,
                "stratified_count": self.stratified_count,
                "files_requiring_review": len(self.files_requiring_review),
            },
            "agreement": {
                "overall_fleiss_kappa": round(self.overall_fleiss_kappa, 4),
                "by_subtype": {k: round(v, 4) for k, v in self.fleiss_kappa_by_subtype.items()},
            },
            "coverage_by_subtype": self.coverage_by_subtype,
            "mandatory_reviews": [r.to_dict() for r in self.mandatory_reviews],
            "stratified_sample": [r.to_dict() for r in self.stratified_sample],
            "files_requiring_review": [f.to_dict() for f in self.files_requiring_review],
        }
