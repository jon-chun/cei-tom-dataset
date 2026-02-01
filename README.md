# CEI Benchmark: Contextual Emotional Inference Dataset

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A benchmark dataset for evaluating pragmatic reasoning in language models, with 300 human-validated scenarios spanning 5 pragmatic subtypes and 3 power relations.

## Quick Start

```python
import json

# Load the benchmark
with open("data/processed/cei_benchmark.json") as f:
    benchmark = json.load(f)

# Each scenario contains:
for scenario in benchmark["scenarios"][:1]:
    print(f"ID: {scenario['scenario_id']}")
    print(f"Context: {scenario['context']}")
    print(f"Utterance: {scenario['utterance']}")
    print(f"Subtype: {scenario['subtype']}")
    print(f"Power Relation: {scenario['power_relation']}")
    print(f"Ground Truth Emotion: {scenario['ground_truth']['emotion']}")
```

## Dataset Overview

| Attribute | Value |
|-----------|-------|
| Total scenarios | 300 |
| Annotations per scenario | 3 |
| Total annotations | 900 |
| Pragmatic subtypes | 5 |
| Power relations | 3 |
| Social domains | 5 |
| Mean annotator agreement | 57% |
| Inter-annotator agreement (Fleiss' κ) | 0.12-0.25 |

### Pragmatic Subtypes

| Subtype | Description | N |
|---------|-------------|---|
| Sarcasm/Irony | Saying the opposite of what is meant | 60 |
| Mixed Signals | Conflicting emotional cues | 60 |
| Strategic Politeness | Polite language masking criticism | 60 |
| Passive Aggression | Indirect hostility through compliance | 60 |
| Deflection/Misdirection | Avoiding uncomfortable topics | 60 |

## Repository Structure

```
cei-tom-dataset/
├── data/
│   ├── gold/              # Raw Label Studio annotations
│   ├── processed/         # Post-QA benchmark with ground truth
│   └── qa_reports/        # Quality assurance analysis
├── src/cei_quality/       # 4-level QA pipeline
├── examples/              # Usage examples
└── config/                # Pipeline configuration
```

## Running the QA Pipeline

```bash
pip install -e .
cei-quality run --config config/config.yml
```

## License

- **Data:** CC-BY-4.0
- **Code:** MIT

## Citation

```bibtex
@article{anonymous2026cei,
  title={CEI: A Benchmark for Evaluating Pragmatic Reasoning in Language Models},
  author={Anonymous},
  journal={arXiv preprint},
  year={2026}
}
```
