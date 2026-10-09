# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CEI (Contextual Emotional Inference) Benchmark is a dataset for evaluating pragmatic reasoning in language models. It contains 300 human-validated scenarios across 5 pragmatic subtypes (sarcasm/irony, mixed signals, strategic politeness, passive aggression, deflection/misdirection) and 3 power relations.

The repository includes:
- Human-annotated gold labels (300 scenarios × 3 annotators)
- A 4-level QA pipeline for annotation quality control
- Pre-computed QA analysis outputs

## Commands

```bash
# Install package in development mode
pip install -e .

# Install with dev dependencies (pytest, ruff, mypy)
pip install -e ".[dev]"

# Run the QA pipeline
cei-quality run --config config/config.yml

# Validate a single annotation file
cei-quality validate path/to/annotations.json

# Generate report from pipeline output
cei-quality report data/qa_reports/sampling_plan.json

# Initialize new project config
cei-quality init

# Linting
ruff check src/
ruff format src/

# Type checking
mypy src/

# Run tests
pytest
```

## Architecture

### QA Pipeline (`src/cei_quality/`)

The pipeline runs 4 validation stages orchestrated by `pipeline.py`:

1. **Stage 1A - Schema Validation** (`validators/schema.py`): Validates JSON structure from Label Studio exports, checks required fields and enum values
2. **Stage 1B - Consistency** (`validators/consistency.py`): Detects duplicates, timing outliers (MAD-based), straight-lining patterns
3. **Stage 1C - Agreement** (`validators/agreement.py`): Computes Fleiss' κ, weighted κ for VAD dimensions, bootstrap confidence intervals
4. **Stage 1D - Plausibility** (`validators/plausibility.py`): Checks VAD-emotion alignment, optional LLM corroboration

Key components:
- `config.py`: Pydantic-based configuration loading with YAML support
- `models.py`: Data models (QualityFlag, AnnotationData, ScenarioData, etc.)
- `loaders.py`: Label Studio JSON parsing
- `scoring.py`: Quality score aggregation
- `sampling.py`: Stratified sampling for human review
- `differentiation.py`: Computes differentiation scores for review prioritization

### Data Structure

- `data/gold/`: Raw Label Studio annotation exports (JSON per annotator per subtype)
- `data/processed/`: Final benchmark with ground truth (`cei_benchmark.json`, `cei_benchmark.csv`)
- `data/qa_reports/`: QA pipeline outputs (agreement metrics, confusion matrices, quality reports)

### Configuration

`config/config.yml` contains:
- Pipeline stage settings (type, mode, API limits)
- Paths for input/output directories
- Quality thresholds (kappa thresholds, timing outliers, review scores)
- Valid label values with numeric mappings for VAD dimensions
- Emotion-VAD profile ranges for plausibility checking
