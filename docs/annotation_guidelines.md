# CEI Benchmark Annotation Guidelines

## Overview

This document describes the annotation protocol used to collect human labels for the CEI (Contextual Emotional Inference) benchmark.

## Task Description

Annotators identify the **listener's** emotional response to pragmatically complex utterances in social contexts. Each scenario contains:

1. **Context**: A situation description setting the scene
2. **Utterance**: What the speaker says
3. **Speaker/Listener roles**: The relationship between participants

## Annotation Targets

### Primary Emotion (Plutchik's Wheel)

Select ONE of Plutchik's 8 basic emotions that best describes the listener's likely emotional response:

| Emotion | Description |
|---------|-------------|
| Joy | Happiness, pleasure, contentment |
| Trust | Acceptance, admiration, openness |
| Fear | Anxiety, apprehension, worry |
| Surprise | Amazement, astonishment, confusion |
| Sadness | Disappointment, grief, pensiveness |
| Disgust | Contempt, loathing, revulsion |
| Anger | Annoyance, frustration, rage |
| Anticipation | Interest, vigilance, expectation |

### VAD Dimensions

Rate each dimension on a 7-point scale:

**Valence** (how pleasant/unpleasant):
- Very unpleasant (-1.0)
- Unpleasant (-0.5)
- Mildly unpleasant (-0.25)
- Neutral (0)
- Mildly pleasant (+0.25)
- Pleasant (+0.5)
- Very pleasant (+1.0)

**Arousal** (how calm/activated):
- Very calm (-1.0)
- Calm (-0.5)
- Mildly calm (-0.25)
- [Neutral] (0)
- Mildly aroused (+0.25)
- Aroused (+0.5)
- Very aroused (+1.0)

**Dominance** (how in-control/controlled):
- Very controlled (-1.0)
- Controlled (-0.5)
- Mildly controlled (-0.25)
- [Neutral] (0)
- Mildly in control (+0.25)
- In control (+0.5)
- Very in control (+1.0)

### Confidence Rating

Rate your confidence in the annotation:
- Very confident
- Mildly confident
- Neutral
- Mildly uncertain
- Very uncertain

## Key Principles

1. **Focus on the listener**: Annotate how the LISTENER likely feels, not the speaker's emotion
2. **Consider pragmatic meaning**: Look beyond literal words to implied meaning
3. **Account for context**: Power relations and social setting affect interpretation
4. **One emotion per scenario**: Choose the single best-fitting primary emotion

## Pragmatic Subtypes

The benchmark includes 5 types of pragmatically complex communication:

| Subtype | What to look for |
|---------|------------------|
| Sarcasm/Irony | Saying the opposite of what is meant |
| Mixed Signals | Conflicting emotional cues |
| Strategic Politeness | Polite language masking criticism |
| Passive Aggression | Indirect hostility through compliance |
| Deflection/Misdirection | Avoiding uncomfortable topics |

## Quality Checks

- Do not straight-line (give identical ratings across scenarios)
- Take time to consider each scenario carefully
- Use the full range of emotion and VAD options
- Flag scenarios that are unclear or ambiguous

## Annotator Training

Annotators complete a qualification round of 20 practice scenarios with feedback before labeling benchmark scenarios.
