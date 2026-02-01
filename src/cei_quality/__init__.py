"""
CEI Quality Pipeline
====================

A comprehensive quality filtering pipeline for the Contextual Emotional Inference (CEI)
benchmark annotation data exported from Label Studio.

This package provides:
- Schema validation for Label Studio exports
- Within-file consistency checking
- Inter-annotator agreement analysis
- LLM-based plausibility verification
- Stratified sampling for human expert review

Example usage::

    from cei_quality import CEIQualityPipeline, load_config

    config = load_config("config/config.yml")
    pipeline = CEIQualityPipeline(config)
    sampling_plan = pipeline.run()

Or via CLI::

    cei-quality run --config config/config.yml --data-dir ./data/

"""

__version__ = "1.0.0"
__author__ = "CEI Research Team"

from cei_quality.config import CEIConfig, load_config
from cei_quality.pipeline import CEIQualityPipeline
from cei_quality.models import (
    QualityFlag,
    QualityIssue,
    RecordQualityReport,
    FileQualityReport,
    SamplingPlan,
)
from cei_quality.sampling import HumanReviewSampler
from cei_quality.comprehensive_report import generate_comprehensive_report

__all__ = [
    # Version
    "__version__",
    # Configuration
    "CEIConfig",
    "load_config",
    # Pipeline
    "CEIQualityPipeline",
    # Models
    "QualityFlag",
    "QualityIssue",
    "RecordQualityReport",
    "FileQualityReport",
    "SamplingPlan",
    # Sampling
    "HumanReviewSampler",
    # Reports
    "generate_comprehensive_report",
]
