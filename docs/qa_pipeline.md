# CEI Quality Assurance Pipeline

## Overview

The CEI QA pipeline performs 4-level validation on human annotations to ensure data quality.

## Pipeline Stages

### Level 1A: Schema Validation

Validates JSON structure from Label Studio exports:
- Required fields present
- Valid enum values for emotions
- Proper data types

### Level 1B: Within-File Consistency

Detects annotation quality issues:
- **Duplicate detection**: Identifies identical annotations
- **Lead time outliers**: Uses MAD-based detection for timing anomalies
- **Straight-lining**: Detects identical ratings across all scenarios

### Level 1C: Inter-Annotator Agreement

Computes agreement metrics:
- **Fleiss' κ**: Overall and by pragmatic subtype
- **Weighted κ**: For ordinal VAD dimensions
- **Bootstrap CIs**: 1000-sample confidence intervals

### Level 1D: Semantic Plausibility

Checks annotation consistency:
- **VAD-Emotion alignment**: Validates that VAD ratings match expected patterns for labeled emotions
- **Conflicting signals**: Flags impossible combinations

## Running the Pipeline

```bash
# Install the package
pip install -e .

# Run full pipeline
cei-quality run --config config/config.yml

# Validate a single file
cei-quality validate --file path/to/annotations.json
```

## Output Reports

The pipeline generates reports in `data/qa_reports/`:

| Report | Contents |
|--------|----------|
| `agreement_metrics.json` | Fleiss' κ values with CIs |
| `confusion_matrix.json` | Human confusion patterns |
| `quality_summary.json` | Overall quality statistics |
| `issues_summary.json` | Flagged scenarios |

## Configuration

Key settings in `config/config.yml`:

```yaml
qa:
  timing:
    mad_threshold: 3.0  # MAD multiplier for outlier detection
  agreement:
    min_kappa: 0.4      # Minimum acceptable kappa
    bootstrap_n: 1000   # Bootstrap iterations
  plausibility:
    vad_tolerance: 0.5  # VAD-emotion mismatch threshold
```

## Interpreting Agreement

| Kappa Range | Interpretation |
|-------------|----------------|
| < 0 | Poor (worse than chance) |
| 0 - 0.20 | Slight |
| 0.21 - 0.40 | Fair |
| 0.41 - 0.60 | Moderate |
| 0.61 - 0.80 | Substantial |
| 0.81 - 1.00 | Almost perfect |

## Quality Flags

Scenarios may be flagged with:
- `TIMING_OUTLIER`: Unusually fast/slow annotation
- `LOW_AGREEMENT`: Annotators disagreed
- `VAD_INCONSISTENT`: VAD doesn't match emotion
- `STRAIGHT_LINE`: Possible inattentive annotator
