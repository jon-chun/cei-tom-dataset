#!/usr/bin/env python3
"""Example: Loading the CEI Benchmark dataset."""

import json
from pathlib import Path


def load_benchmark():
    """Load the processed CEI benchmark with ground truth labels."""
    data_dir = Path(__file__).parent.parent / "data" / "processed"

    with open(data_dir / "cei_benchmark.json") as f:
        benchmark = json.load(f)

    print(f"Loaded {len(benchmark['scenarios'])} scenarios")
    print(f"Metadata: {benchmark['metadata']}")

    return benchmark


def explore_scenario(scenario: dict):
    """Print details of a single scenario."""
    print(f"\n{'='*60}")
    print(f"Scenario ID: {scenario['scenario_id']}")
    print(f"Subtype: {scenario['subtype']}")
    print(f"Power Relation: {scenario['power_relation']}")
    print(f"Domain: {scenario['domain']}")
    print(f"\nContext:\n  {scenario['context']}")
    print(f"\nUtterance:\n  \"{scenario['utterance']}\"")
    print(f"\nGround Truth:")
    gt = scenario['ground_truth']
    print(f"  Emotion: {gt['emotion']}")
    print(f"  VAD: V={gt['valence']:.2f}, A={gt['arousal']:.2f}, D={gt['dominance']:.2f}")
    print(f"  Agreement: {gt['annotator_agreement']:.0%}")
    print(f"{'='*60}")


def main():
    benchmark = load_benchmark()

    # Show distribution by subtype
    subtypes = {}
    for s in benchmark['scenarios']:
        subtypes[s['subtype']] = subtypes.get(s['subtype'], 0) + 1

    print("\nDistribution by subtype:")
    for subtype, count in sorted(subtypes.items()):
        print(f"  {subtype}: {count}")

    # Show distribution by power relation
    powers = {}
    for s in benchmark['scenarios']:
        powers[s['power_relation']] = powers.get(s['power_relation'], 0) + 1

    print("\nDistribution by power relation:")
    for power, count in sorted(powers.items()):
        print(f"  {power}: {count}")

    # Show example scenarios
    print("\n\nExample scenarios:")
    for scenario in benchmark['scenarios'][:2]:
        explore_scenario(scenario)


if __name__ == "__main__":
    main()
