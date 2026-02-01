"""
Configuration management for CEI Quality Pipeline.

This module provides:
- Pydantic models for type-safe configuration
- YAML configuration loading
- Environment variable overrides
- Configuration validation

Configuration is loaded from YAML files and can be overridden via:
1. Environment variables (CEI_<SECTION>_<KEY>)
2. Command-line arguments
3. Local config file (config/config.local.yml)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# =============================================================================
# Configuration Models
# =============================================================================


class ProjectConfig(BaseModel):
    """Project metadata configuration."""

    name: str = "CEI Benchmark"
    version: str = "1.0.0"
    description: str = "Contextual Emotional Inference Benchmark Quality Pipeline"


class PathsConfig(BaseModel):
    """Path configuration for input/output directories."""

    data_dir: Path = Path("scenarios/openai/gpt-5-mini/gold-gpt5mini")
    output_dir: Path = Path("outputs/quality")
    reports_dir: Path = Path("reports")
    logs_dir: Path = Path("logs")
    cache_dir: Path = Path(".cache/cei_quality")

    @field_validator("data_dir", "output_dir", "reports_dir", "logs_dir", "cache_dir", mode="before")
    @classmethod
    def convert_to_path(cls, v: Any) -> Path:
        """Convert string paths to Path objects."""
        return Path(v) if isinstance(v, str) else v

    def resolve_paths(self, project_root: Path) -> "PathsConfig":
        """Resolve relative paths against project root."""

        def resolve(p: Path) -> Path:
            return p if p.is_absolute() else project_root / p

        return PathsConfig(
            data_dir=resolve(self.data_dir),
            output_dir=resolve(self.output_dir),
            reports_dir=resolve(self.reports_dir),
            logs_dir=resolve(self.logs_dir),
            cache_dir=resolve(self.cache_dir),
        )


class SchemaConfig(BaseModel):
    """Data schema configuration."""

    scenarios_per_subtype: int = 60
    annotators_per_subtype: int = 3
    total_subtypes: int = 5
    scenario_id_field: str = "id"
    file_extensions: List[str] = [".json", ".jsonl"]
    subtypes: List[str] = [
        "sarcasm-irony",
        "mixed-signals",
        "strategic-politeness",
        "passive-aggression",
        "deflection-misdirection",
    ]


class LabelStudioFieldsConfig(BaseModel):
    """Label Studio field mapping configuration."""

    class DataFields(BaseModel):
        situation: str = "sd_situation"
        utterance: str = "sd_utterance"
        speaker_role: str = "sd_speaker_role"
        listener_role: str = "sd_listener_role"
        scenario_id: str = "id"

    class LabelFields(BaseModel):
        plutchik_emotion: str = "sl_plutchik_primary"
        valence: str = "sl_v"
        arousal: str = "sl_a"
        dominance: str = "sl_d"
        confidence: str = "sl_confidence"

    data_fields: DataFields = Field(default_factory=DataFields)
    label_fields: LabelFields = Field(default_factory=LabelFields)


class LabelValue(BaseModel):
    """A single valid label value with numeric mapping."""

    value: str
    numeric: float


class ValidLabelsConfig(BaseModel):
    """Valid label values configuration."""

    plutchik_emotions: List[str] = [
        "joy",
        "trust",
        "fear",
        "surprise",
        "sadness",
        "disgust",
        "anger",
        "anticipation",
    ]
    valence: List[LabelValue] = []
    arousal: List[LabelValue] = []
    dominance: List[LabelValue] = []
    confidence: List[LabelValue] = []

    def get_numeric_map(self, dimension: str) -> Dict[str, float]:
        """Get string-to-numeric mapping for a VAD dimension."""
        values: List[LabelValue] = getattr(self, dimension, [])
        return {v.value.lower(): v.numeric for v in values}

    def get_valid_values(self, dimension: str) -> set[str]:
        """Get set of valid string values for a dimension."""
        if dimension == "plutchik_emotions":
            return set(self.plutchik_emotions)
        values: List[LabelValue] = getattr(self, dimension, [])
        return {v.value.lower() for v in values}


class VADRange(BaseModel):
    """Expected VAD range for an emotion."""

    valence: tuple[float, float]
    arousal: tuple[float, float]
    dominance: tuple[float, float]

    @field_validator("valence", "arousal", "dominance", mode="before")
    @classmethod
    def convert_list_to_tuple(cls, v: Any) -> tuple[float, float]:
        """Convert list to tuple."""
        if isinstance(v, list):
            return (float(v[0]), float(v[1]))
        return v


class QualityConfig(BaseModel):
    """Quality scoring configuration."""

    class SeverityWeights(BaseModel):
        critical: float = 0.40
        major: float = 0.20
        minor: float = 0.05
        info: float = 0.00

    class StageWeights(BaseModel):
        schema: float = 0.25
        consistency: float = 0.20
        agreement: float = 0.35
        plausibility: float = 0.20

    class ScoreThresholds(BaseModel):
        excellent: float = 0.95
        good: float = 0.85
        acceptable: float = 0.70
        questionable: float = 0.50

    class LeadTimeConfig(BaseModel):
        impossibly_fast: float = 3.0
        suspiciously_fast: float = 5.0
        unusually_slow: float = 600.0

    class VADDisagreementConfig(BaseModel):
        high: float = 1.5
        moderate: float = 1.0

    severity_weights: SeverityWeights = Field(default_factory=SeverityWeights)
    stage_weights: StageWeights = Field(default_factory=StageWeights)
    score_thresholds: ScoreThresholds = Field(default_factory=ScoreThresholds)
    lead_time: LeadTimeConfig = Field(default_factory=LeadTimeConfig)
    vad_disagreement: VADDisagreementConfig = Field(default_factory=VADDisagreementConfig)
    vad_tolerance: float = 0.5


class ThresholdsConfig(BaseModel):
    """Quality thresholds for differentiation."""

    # Lead time thresholds (seconds)
    min_lead_time_seconds: float = 5.0
    impossibly_fast_seconds: float = 3.0
    unusually_slow_seconds: float = 600.0

    # Dwell time outlier detection (z-score)
    dwell_time_outlier_z: float = 2.0

    # Inter-annotator agreement thresholds
    fleiss_kappa_warning: float = 0.3
    fleiss_kappa_acceptable: float = 0.4
    fleiss_kappa_good: float = 0.6

    # Quality score thresholds
    mandatory_review_score: float = 0.80
    flag_review_score: float = 0.90


class ReviewConfig(BaseModel):
    """Human review settings."""

    # Show all annotators when reviewing a flagged scenario
    show_all_annotators: bool = True

    # Maximum items in review queue
    max_review_items: int = 100

    # Priority thresholds (1-100 scale, higher = more urgent)
    priority_critical: int = 90
    priority_high: int = 70
    priority_medium: int = 50


class SamplingConfig(BaseModel):
    """Human review sampling configuration."""

    random_seed: int = 42
    mandatory_priority_threshold: int = 8
    stratified_rate: float = 0.15
    min_per_subtype: int = 3
    mandatory_score_threshold: float = 0.50
    file_issue_rate_threshold: float = 0.20


class LLMConfig(BaseModel):
    """LLM configuration for plausibility checks."""

    enabled: bool = False
    provider: str = "openai"
    model: str = "gpt-5-mini"
    temperature: float = 0.1
    batch_size: int = 5
    max_retries: int = 3
    timeout: int = 60


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_logging: bool = True
    log_file_pattern: str = "cei_quality_{timestamp}.log"
    rich_console: bool = True


class OutputConfig(BaseModel):
    """Output configuration."""

    class GenerateConfig(BaseModel):
        sampling_plan: bool = True
        all_reports: bool = True
        file_reports: bool = True
        issues_summary: bool = True
        review_queue: bool = True
        text_report: bool = True

    overwrite: bool = False
    json_indent: int = 2
    generate: GenerateConfig = Field(default_factory=GenerateConfig)
    review_queue_max_items: int = 100


# =============================================================================
# Main Configuration Class
# =============================================================================


class CEIConfig(BaseSettings):
    """
    Main configuration class for CEI Quality Pipeline.

    Loads configuration from YAML files with environment variable overrides.
    Environment variables use the prefix CEI_ and nested keys are separated
    by double underscores (e.g., CEI_PATHS__DATA_DIR).
    """

    model_config = SettingsConfigDict(
        env_prefix="CEI_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    project: ProjectConfig = Field(default_factory=ProjectConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    schema_config: SchemaConfig = Field(default_factory=SchemaConfig, alias="schema")
    labelstudio: LabelStudioFieldsConfig = Field(default_factory=LabelStudioFieldsConfig)
    valid_labels: ValidLabelsConfig = Field(default_factory=ValidLabelsConfig)
    emotion_vad_profiles: Dict[str, VADRange] = Field(default_factory=dict)
    quality: QualityConfig = Field(default_factory=QualityConfig)
    thresholds: ThresholdsConfig = Field(default_factory=ThresholdsConfig)
    review: ReviewConfig = Field(default_factory=ReviewConfig)
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)

    # Runtime attributes (not from config file)
    _project_root: Optional[Path] = None
    _config_path: Optional[Path] = None

    @property
    def project_root(self) -> Path:
        """Get the project root directory."""
        if self._project_root is not None:
            return self._project_root
        # Default to current directory
        return Path.cwd()

    @project_root.setter
    def project_root(self, value: Path) -> None:
        """Set the project root directory."""
        self._project_root = value

    def get_resolved_paths(self) -> PathsConfig:
        """Get paths resolved against project root."""
        return self.paths.resolve_paths(self.project_root)

    def get_vad_numeric(self, dimension: str, value: str) -> float:
        """Get numeric value for a VAD dimension."""
        mapping = self.valid_labels.get_numeric_map(dimension)
        return mapping.get(value.lower().strip(), 0.0)

    def get_emotion_profile(self, emotion: str) -> Optional[VADRange]:
        """Get expected VAD profile for an emotion."""
        return self.emotion_vad_profiles.get(emotion.lower())

    def is_valid_label(self, field: str, value: str) -> bool:
        """Check if a label value is valid for a field."""
        valid = self.valid_labels.get_valid_values(field)
        return value.lower().strip() in valid


# =============================================================================
# Configuration Loading
# =============================================================================


def load_config(
    config_path: Optional[Union[str, Path]] = None,
    project_root: Optional[Union[str, Path]] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> CEIConfig:
    """
    Load configuration from YAML file with optional overrides.

    Args:
        config_path: Path to config YAML file. If None, looks for config/config.yml
        project_root: Project root directory. If None, uses current directory
        overrides: Dictionary of config overrides (dotted keys supported)

    Returns:
        Loaded and validated CEIConfig instance

    Example:
        >>> config = load_config("config/config.yml")
        >>> config = load_config(overrides={"llm.enabled": True})
    """
    # Determine project root
    if project_root is not None:
        root = Path(project_root)
    else:
        root = Path.cwd()

    # Determine config path
    if config_path is not None:
        cfg_path = Path(config_path)
        if not cfg_path.is_absolute():
            cfg_path = root / cfg_path
    else:
        # Look for default locations
        candidates = [
            root / "config" / "config.yml",
            root / "config" / "config.yaml",
            root / "config.yml",
            root / "config.yaml",
        ]
        cfg_path = None
        for candidate in candidates:
            if candidate.exists():
                cfg_path = candidate
                break

        if cfg_path is None:
            # No config file found, use defaults (with overrides if provided)
            if overrides:
                config_data = _apply_overrides({}, overrides)
                config = CEIConfig(**config_data)
            else:
                config = CEIConfig()
            config._project_root = root
            return config

    # Load YAML config
    config_data: Dict[str, Any] = {}
    if cfg_path.exists():
        with open(cfg_path, "r") as f:
            config_data = yaml.safe_load(f) or {}

    # Check for local overrides file
    local_cfg_path = cfg_path.parent / "config.local.yml"
    if local_cfg_path.exists():
        with open(local_cfg_path, "r") as f:
            local_data = yaml.safe_load(f) or {}
            config_data = _deep_merge(config_data, local_data)

    # Apply programmatic overrides
    if overrides:
        config_data = _apply_overrides(config_data, overrides)

    # Convert emotion_vad_profiles to proper format
    if "emotion_vad_profiles" in config_data:
        profiles = config_data["emotion_vad_profiles"]
        for emotion, ranges in profiles.items():
            if isinstance(ranges, dict):
                for dim in ["valence", "arousal", "dominance"]:
                    if dim in ranges and isinstance(ranges[dim], list):
                        ranges[dim] = tuple(ranges[dim])

    # Create config instance
    config = CEIConfig(**config_data)
    config._project_root = root
    config._config_path = cfg_path

    return config


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge two dictionaries, with override taking precedence."""
    result = base.copy()

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value

    return result


def _apply_overrides(config: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Apply dotted-key overrides to config dictionary."""
    result = config.copy()

    for key, value in overrides.items():
        parts = key.split(".")
        current = result

        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]

        current[parts[-1]] = value

    return result


def _convert_paths_to_strings(obj: Any) -> Any:
    """Recursively convert Path objects to strings for YAML serialization."""
    if isinstance(obj, Path):
        return str(obj)
    elif isinstance(obj, dict):
        return {k: _convert_paths_to_strings(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_paths_to_strings(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(_convert_paths_to_strings(v) for v in obj)
    return obj


def save_config(config: CEIConfig, path: Union[str, Path]) -> None:
    """
    Save configuration to YAML file.

    Args:
        config: CEIConfig instance to save
        path: Output path for YAML file
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to dictionary, excluding runtime attributes
    data = config.model_dump(exclude={"_project_root", "_config_path"})

    # Convert Path objects to strings for YAML compatibility
    data = _convert_paths_to_strings(data)

    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
