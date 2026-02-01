#!/usr/bin/env python3
"""Generate the processed benchmark file from gold annotations."""

import json
import re
from collections import Counter
from pathlib import Path


# VAD scale mappings
VAD_SCALES = {
    # Valence
    "very unpleasant": -1.0,
    "unpleasant": -0.5,
    "mildly unpleasant": -0.25,
    "neutral": 0.0,
    "mildly pleasant": 0.25,
    "pleasant": 0.5,
    "very pleasant": 1.0,
    # Arousal
    "very calm": -1.0,
    "calm": -0.5,
    "mildly calm": -0.25,
    "mildly aroused": 0.25,
    "aroused": 0.5,
    "very aroused": 1.0,
    # Dominance
    "very controlled": -1.0,
    "controlled": -0.5,
    "mildly controlled": -0.25,
    "mildly in control": 0.25,
    "in control": 0.5,
    "very in control": 1.0,
}

EMOTIONS = ["anger", "anticipation", "disgust", "fear", "joy", "sadness", "surprise", "trust"]

SUBTYPE_MAPPING = {
    "sarcasm-irony": "sarcasm-irony",
    "mixed-signals": "mixed-signals",
    "passive-aggression": "passive-aggression",
    "deflection-misdirection": "deflection-misdirection",
    "strategic-politeness": "strategic-politeness",
}


def extract_subtype_from_filename(filename: str) -> str:
    """Extract pragmatic subtype from annotation filename."""
    name = filename.lower()
    for key, value in SUBTYPE_MAPPING.items():
        if key.replace("-", "") in name.replace("-", "").replace("_", ""):
            return value
    return "unknown"


def extract_power_relation(speaker_role: str, listener_role: str) -> str:
    """Infer power relation from speaker/listener roles."""
    high_power = {"boss", "manager", "supervisor", "director", "executive", "ceo", "principal",
                  "professor", "doctor", "judge", "parent", "teacher", "coach", "captain"}
    low_power = {"employee", "worker", "intern", "student", "child", "subordinate", "assistant",
                 "patient", "defendant"}

    speaker_lower = speaker_role.lower()
    listener_lower = listener_role.lower()

    speaker_is_high = any(p in speaker_lower for p in high_power)
    speaker_is_low = any(p in speaker_lower for p in low_power)
    listener_is_high = any(p in listener_lower for p in high_power)
    listener_is_low = any(p in listener_lower for p in low_power)

    if speaker_is_high and listener_is_low:
        return "high-to-low"
    elif speaker_is_low and listener_is_high:
        return "low-to-high"
    else:
        return "peer-to-peer"


def extract_domain(situation: str, speaker_role: str, listener_role: str) -> str:
    """Infer social domain from scenario content."""
    text = f"{situation} {speaker_role} {listener_role}".lower()

    if any(w in text for w in ["office", "meeting", "manager", "employee", "work", "project",
                                "deadline", "boss", "colleague", "coworker", "team"]):
        return "workplace"
    elif any(w in text for w in ["family", "parent", "child", "sibling", "home", "dinner",
                                  "holiday", "wedding", "brother", "sister", "mother", "father"]):
        return "family"
    elif any(w in text for w in ["friend", "party", "hangout", "bar", "restaurant", "movie",
                                  "vacation", "trip", "concert"]):
        return "social"
    elif any(w in text for w in ["doctor", "patient", "hospital", "clinic", "nurse", "medical",
                                  "customer", "service", "store", "shop", "hotel"]):
        return "service"
    else:
        return "general"


def parse_annotation_result(result: list) -> dict:
    """Parse Label Studio annotation result into structured data."""
    parsed = {
        "emotion": None,
        "valence": None,
        "arousal": None,
        "dominance": None,
        "confidence": None,
    }

    for item in result:
        from_name = item.get("from_name", "")
        choices = item.get("value", {}).get("choices", [])
        if not choices:
            continue

        choice = choices[0].lower()

        if from_name == "sl_plutchik_primary":
            if choice in EMOTIONS:
                parsed["emotion"] = choice
        elif from_name == "sl_v":
            parsed["valence"] = VAD_SCALES.get(choice, 0.0)
        elif from_name == "sl_a":
            parsed["arousal"] = VAD_SCALES.get(choice, 0.0)
        elif from_name == "sl_d":
            parsed["dominance"] = VAD_SCALES.get(choice, 0.0)
        elif from_name == "sl_confidence":
            parsed["confidence"] = choice

    return parsed


def compute_ground_truth(annotations: list) -> dict:
    """Compute ground truth from multiple annotations via majority vote."""
    emotions = [a["emotion"] for a in annotations if a["emotion"]]
    valences = [a["valence"] for a in annotations if a["valence"] is not None]
    arousals = [a["arousal"] for a in annotations if a["arousal"] is not None]
    dominances = [a["dominance"] for a in annotations if a["dominance"] is not None]

    # Majority vote for emotion
    if emotions:
        emotion_counts = Counter(emotions)
        majority_emotion = emotion_counts.most_common(1)[0][0]
        agreement = emotion_counts[majority_emotion] / len(emotions)
    else:
        majority_emotion = None
        agreement = 0.0

    # Mean for VAD
    mean_valence = sum(valences) / len(valences) if valences else 0.0
    mean_arousal = sum(arousals) / len(arousals) if arousals else 0.0
    mean_dominance = sum(dominances) / len(dominances) if dominances else 0.0

    return {
        "emotion": majority_emotion,
        "valence": round(mean_valence, 3),
        "arousal": round(mean_arousal, 3),
        "dominance": round(mean_dominance, 3),
        "annotator_agreement": round(agreement, 3),
        "n_annotators": len(annotations),
    }


def load_all_gold_annotations(gold_dir: Path) -> list:
    """Load all gold annotation files and aggregate scenarios by content."""
    # Group annotations by scenario content (situation + utterance)
    scenario_annotations = {}  # key: (subtype, situation, utterance) -> list of annotations

    for json_file in sorted(gold_dir.glob("*.json")):
        subtype = extract_subtype_from_filename(json_file.name)

        with open(json_file) as f:
            tasks = json.load(f)

        for task in tasks:
            data = task.get("data", {})
            annotations_raw = task.get("annotations", [])

            situation = data.get("sd_situation", "")
            utterance = data.get("sd_utterance", "")
            speaker_role = data.get("sd_speaker_role", "")
            listener_role = data.get("sd_listener_role", "")

            # Create unique key for this scenario
            key = (subtype, situation, utterance)

            if key not in scenario_annotations:
                scenario_annotations[key] = {
                    "subtype": subtype,
                    "situation": situation,
                    "utterance": utterance,
                    "speaker_role": speaker_role,
                    "listener_role": listener_role,
                    "annotations": [],
                    "source_files": [],
                }

            scenario_annotations[key]["source_files"].append(json_file.name)

            # Parse each annotation
            for ann in annotations_raw:
                result = ann.get("result", [])
                if result:
                    parsed = parse_annotation_result(result)
                    if parsed["emotion"]:
                        scenario_annotations[key]["annotations"].append(parsed)

    # Convert to final scenario list
    scenarios = []
    scenario_id = 1

    for key, data in sorted(scenario_annotations.items()):
        if not data["annotations"]:
            continue

        # Compute ground truth from all annotations for this scenario
        ground_truth = compute_ground_truth(data["annotations"])

        power_relation = extract_power_relation(data["speaker_role"], data["listener_role"])
        domain = extract_domain(data["situation"], data["speaker_role"], data["listener_role"])

        scenario = {
            "scenario_id": f"CEI-{scenario_id:04d}",
            "subtype": data["subtype"],
            "power_relation": power_relation,
            "domain": domain,
            "context": data["situation"],
            "utterance": data["utterance"],
            "speaker_role": data["speaker_role"],
            "listener_role": data["listener_role"],
            "ground_truth": ground_truth,
        }
        scenarios.append(scenario)
        scenario_id += 1

    return scenarios


def compute_dataset_statistics(scenarios: list) -> dict:
    """Compute summary statistics for the benchmark."""
    n_scenarios = len(scenarios)

    # By subtype
    by_subtype = Counter(s["subtype"] for s in scenarios)

    # By power relation
    by_power = Counter(s["power_relation"] for s in scenarios)

    # By domain
    by_domain = Counter(s["domain"] for s in scenarios)

    # Agreement distribution
    agreements = [s["ground_truth"]["annotator_agreement"] for s in scenarios]
    mean_agreement = sum(agreements) / len(agreements) if agreements else 0

    # Emotion distribution
    by_emotion = Counter(s["ground_truth"]["emotion"] for s in scenarios if s["ground_truth"]["emotion"])

    return {
        "n_scenarios": n_scenarios,
        "n_annotations": sum(s["ground_truth"]["n_annotators"] for s in scenarios),
        "by_subtype": dict(by_subtype),
        "by_power_relation": dict(by_power),
        "by_domain": dict(by_domain),
        "by_emotion": dict(by_emotion),
        "mean_annotator_agreement": round(mean_agreement, 3),
    }


def main():
    repo_root = Path(__file__).parent.parent
    gold_dir = repo_root / "data" / "gold"
    output_dir = repo_root / "data" / "processed"
    output_dir.mkdir(exist_ok=True)

    print("Loading gold annotations...")
    scenarios = load_all_gold_annotations(gold_dir)
    print(f"Loaded {len(scenarios)} scenarios")

    print("\nComputing statistics...")
    statistics = compute_dataset_statistics(scenarios)

    print(f"\nStatistics:")
    print(f"  Total scenarios: {statistics['n_scenarios']}")
    print(f"  Total annotations: {statistics['n_annotations']}")
    print(f"  Mean agreement: {statistics['mean_annotator_agreement']:.1%}")
    print(f"\n  By subtype:")
    for subtype, count in sorted(statistics['by_subtype'].items()):
        print(f"    {subtype}: {count}")
    print(f"\n  By power relation:")
    for power, count in sorted(statistics['by_power_relation'].items()):
        print(f"    {power}: {count}")

    # Create benchmark output
    benchmark = {
        "metadata": {
            "name": "CEI Benchmark",
            "version": "1.0.0",
            "description": "Contextual Emotional Inference benchmark for pragmatic reasoning",
            "n_scenarios": statistics["n_scenarios"],
            "n_annotations": statistics["n_annotations"],
            "pragmatic_subtypes": list(statistics["by_subtype"].keys()),
            "power_relations": list(statistics["by_power_relation"].keys()),
            "social_domains": list(statistics["by_domain"].keys()),
            "emotions": EMOTIONS,
        },
        "statistics": statistics,
        "scenarios": scenarios,
    }

    # Write JSON
    output_file = output_dir / "cei_benchmark.json"
    with open(output_file, "w") as f:
        json.dump(benchmark, f, indent=2)
    print(f"\nWrote benchmark to: {output_file}")

    # Write CSV for convenience
    import csv
    csv_file = output_dir / "cei_benchmark.csv"
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "scenario_id", "subtype", "power_relation", "domain",
            "context", "utterance", "speaker_role", "listener_role",
            "emotion", "valence", "arousal", "dominance", "agreement"
        ])
        for s in scenarios:
            gt = s["ground_truth"]
            writer.writerow([
                s["scenario_id"], s["subtype"], s["power_relation"], s["domain"],
                s["context"], s["utterance"], s["speaker_role"], s["listener_role"],
                gt["emotion"], gt["valence"], gt["arousal"], gt["dominance"],
                gt["annotator_agreement"]
            ])
    print(f"Wrote CSV to: {csv_file}")

    print("\nDone!")


if __name__ == "__main__":
    main()
