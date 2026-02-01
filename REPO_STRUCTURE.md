# CEI Dataset Repository Structure

**Purpose:** Minimal, clean repository for the CEI Benchmark dataset paper (ArXiv/DMLR).

**What's Included:**
- Human-annotated gold labels (300 scenarios × 3 annotators)
- 4-level QA pipeline code
- Pre-computed QA analysis outputs
- Evaluation framework and examples

**What's Excluded (to protect companion paper novelty):**
- LLM response data
- Model evaluation code
- Intervention experiments
- Power-stratified model analysis
- Compositional generalization tests

---

## Directory Structure

```
cei-tom-dataset/
├── README.md                    # Dataset overview, usage, citation
├── LICENSE                      # MIT for code, CC-BY-4.0 for data
├── pyproject.toml              # Python package definition
├── config/
│   └── config.yml              # QA pipeline configuration
├── data/
│   ├── gold/                   # Human annotations (Label Studio JSON)
│   │   ├── scenarios.json      # Aggregated scenario data
│   │   └── annotations/        # Per-annotator files
│   ├── processed/              # Post-QA processed data
│   │   ├── cei_benchmark.json  # Final benchmark with ground truth
│   │   ├── cei_benchmark.csv   # CSV format for convenience
│   │   └── splits/             # Train/val/test splits
│   └── qa_reports/             # QA pipeline outputs
│       ├── agreement_metrics.json
│       ├── confusion_matrix.json
│       └── quality_summary.json
├── src/
│   └── cei_quality/            # QA pipeline code
│       ├── __init__.py
│       ├── cli.py              # Command-line interface
│       ├── pipeline.py         # Main pipeline orchestration
│       ├── config.py           # Configuration handling
│       ├── models.py           # Data models (QualityFlag, etc.)
│       ├── loaders.py          # Data loading utilities
│       ├── scoring.py          # Quality scoring
│       ├── sampling.py         # Stratified sampling
│       └── validators/         # 4-level QA validators
│           ├── __init__.py
│           ├── schema.py       # Level 1A: Schema validation
│           ├── consistency.py  # Level 1B: Within-file consistency
│           ├── agreement.py    # Level 1C: Inter-annotator agreement
│           └── plausibility.py # Level 1D: Semantic plausibility
├── examples/
│   ├── load_dataset.py         # How to load the dataset
│   ├── run_qa_pipeline.py      # How to run QA
│   └── compute_statistics.py   # How to compute paper statistics
└── docs/
    ├── annotation_guidelines.md
    ├── qa_pipeline.md
    └── evaluation_protocol.md
```

---

## What Each Directory Contains

### `data/gold/`
Raw Label Studio exports with full annotation metadata (timing, confidence, notes).

### `data/processed/`
Post-QA dataset with:
- Ground truth labels (majority vote + expert adjudication)
- Scenario metadata (subtype, power relation, domain)
- Quality flags from QA pipeline

### `data/qa_reports/`
Pre-computed QA analysis:
- Fleiss' κ by subtype
- Human confusion matrix
- Quality flag distribution

### `src/cei_quality/`
The 4-level QA pipeline code:
1. Schema validation
2. Within-file consistency (MAD-based timing outliers, straight-lining)
3. Inter-annotator agreement (Fleiss' κ, weighted κ for VAD)
4. Semantic plausibility (VAD-emotion consistency)

---

*This structure supports the dataset paper without revealing model evaluation results reserved for the companion cognitive science paper.*
