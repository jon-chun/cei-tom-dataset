# CEI Benchmark Evaluation Protocol

## Overview

This document describes how to evaluate models on the CEI benchmark.

## Task Format

For each scenario, models must:
1. Read the context and utterance
2. Predict the listener's primary emotion (1 of 8)
3. Optionally predict VAD dimensions

## Input Format

```json
{
  "scenario_id": "CEI-0001",
  "context": "A situation description...",
  "utterance": "What the speaker says",
  "speaker_role": "boss",
  "listener_role": "employee"
}
```

## Output Format

```json
{
  "scenario_id": "CEI-0001",
  "predicted_emotion": "sadness",
  "predicted_valence": -0.5,
  "predicted_arousal": 0.25,
  "predicted_dominance": -0.25
}
```

## Evaluation Metrics

### Primary Metrics

| Metric | Description |
|--------|-------------|
| Accuracy | Exact match with ground truth emotion |
| Macro-F1 | F1 averaged across 8 emotion classes |
| Weighted-F1 | F1 weighted by class frequency |

### VAD Metrics

| Metric | Description |
|--------|-------------|
| MAE | Mean absolute error for each dimension |
| Pearson r | Correlation with human ratings |

### Stratified Analysis

Report metrics separately for:
- Each pragmatic subtype (5 categories)
- Each power relation (3 categories)
- Each domain (4 categories)

## Baseline Performance

### Human Performance

From 3-annotator validation:
- Overall agreement rate: ~57%
- Fleiss' κ: 0.12-0.25 depending on subtype
- Highest agreement: Sarcasm/Irony (23%)
- Lowest agreement: Passive Aggression (9%)

### Difficulty Distribution

Based on annotator agreement:
- Easy (≥67% agreement): ~15% of scenarios
- Medium (34-66% agreement): ~40% of scenarios
- Hard (<34% agreement): ~45% of scenarios

## Prompt Templates

### Zero-shot Template

```
Context: {context}

The speaker ({speaker_role}) says to the listener ({listener_role}):
"{utterance}"

What emotion is the listener most likely feeling?
Choose from: anger, anticipation, disgust, fear, joy, sadness, surprise, trust

Answer:
```

### Chain-of-Thought Template

```
Context: {context}

The speaker ({speaker_role}) says to the listener ({listener_role}):
"{utterance}"

Step 1: What is the literal meaning of the utterance?
Step 2: What pragmatic meaning might be implied?
Step 3: How might the listener interpret this?
Step 4: What emotion is the listener most likely feeling?

Choose from: anger, anticipation, disgust, fear, joy, sadness, surprise, trust
```

## Submission Format

For benchmark submissions, provide:
1. Model predictions in JSON format
2. Model card with architecture details
3. Prompt template used
4. Number of parameters / API used

## Citation

```bibtex
@article{anonymous2026cei,
  title={CEI: A Benchmark for Evaluating Pragmatic Reasoning},
  author={Anonymous},
  year={2026}
}
```
