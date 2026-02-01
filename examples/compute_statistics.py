#!/usr/bin/env python3
"""Example: Computing statistics for the CEI Benchmark paper."""

import json
from pathlib import Path
from collections import defaultdict


def load_qa_reports():
    """Load pre-computed QA reports."""
    qa_dir = Path(__file__).parent.parent / "data" / "qa_reports"

    reports = {}
    for report_file in qa_dir.glob("*.json"):
        with open(report_file) as f:
            reports[report_file.stem] = json.load(f)

    return reports


def compute_agreement_stats(reports: dict):
    """Compute inter-annotator agreement statistics."""
    agreement = reports.get("agreement_metrics", {})

    print("\n" + "=" * 60)
    print("INTER-ANNOTATOR AGREEMENT (Fleiss' Kappa)")
    print("=" * 60)

    # Overall kappa
    overall_kappa = agreement.get("fleiss_kappa", {}).get("overall", 0)
    print(f"\nOverall Fleiss' κ: {overall_kappa:.3f}")

    # By subtype
    print("\nBy Pragmatic Subtype:")
    kappa_by_subtype = agreement.get("fleiss_kappa", {}).get("by_subtype", {})
    for subtype, kappa in sorted(kappa_by_subtype.items()):
        interp = agreement.get("interpretation", {}).get(subtype, "")
        print(f"  {subtype}: κ = {kappa:.3f} ({interp})")

    # Weighted kappa for VAD
    print("\nWeighted Kappa for VAD Dimensions:")
    vad_kappa = agreement.get("weighted_kappa_vad", {})
    for subtype, dims in sorted(vad_kappa.items()):
        v, a, d = dims.get("valence", 0), dims.get("arousal", 0), dims.get("dominance", 0)
        print(f"  {subtype}: V={v:.3f}, A={a:.3f}, D={d:.3f}")


def compute_confusion_stats(reports: dict):
    """Compute human confusion matrix statistics."""
    confusion = reports.get("confusion_matrix", {})

    print("\n" + "=" * 60)
    print("HUMAN CONFUSION PATTERNS")
    print("=" * 60)

    overall = confusion.get("overall", {})
    total_pairs = overall.get("total_pairs", 0)
    agreement_count = overall.get("agreement_count", 0)
    agreement_rate = overall.get("agreement_rate", 0)

    print(f"\nTotal annotation pairs: {total_pairs}")
    print(f"Agreement count: {agreement_count}")
    print(f"Agreement rate: {agreement_rate:.1%}")

    # By subtype
    print("\nAgreement Rate by Subtype:")
    by_subtype = confusion.get("by_subtype", {})
    for subtype, data in sorted(by_subtype.items()):
        rate = data.get("agreement_rate", 0)
        print(f"  {subtype}: {rate:.1%}")

    # Most common confusions
    print("\nTop 5 Overall Confusion Pairs:")
    if "overall" in confusion:
        matrix = confusion["overall"].get("matrix", {})
        confusions = []
        for e1 in matrix:
            for e2 in matrix[e1]:
                if e1 < e2:  # Avoid counting twice
                    confusions.append((e1, e2, matrix[e1][e2] + matrix[e2][e1]))
        confusions.sort(key=lambda x: x[2], reverse=True)
        for e1, e2, count in confusions[:5]:
            print(f"  {e1} <-> {e2}: {count}")


def compute_subtype_difficulty(reports: dict):
    """Compute difficulty ranking by subtype based on human agreement."""
    confusion = reports.get("confusion_matrix", {})

    print("\n" + "=" * 60)
    print("SCENARIO DIFFICULTY (by Human Agreement)")
    print("=" * 60)

    by_subtype = confusion.get("by_subtype", {})
    difficulties = []
    for subtype, data in by_subtype.items():
        rate = data.get("agreement_rate", 0)
        difficulties.append((subtype, rate))

    difficulties.sort(key=lambda x: x[1], reverse=True)

    print("\nRanked by human agreement (higher = easier):")
    for i, (subtype, rate) in enumerate(difficulties, 1):
        difficulty = "Easier" if rate > 0.2 else ("Medium" if rate > 0.15 else "Harder")
        print(f"  {i}. {subtype}: {rate:.1%} ({difficulty})")


def main():
    print("CEI Benchmark Statistics")
    print("=" * 60)

    reports = load_qa_reports()

    if not reports:
        print("No QA reports found. Run the QA pipeline first:")
        print("  cei-quality run --config config/config.yml")
        return

    print(f"Loaded {len(reports)} QA reports")

    compute_agreement_stats(reports)
    compute_confusion_stats(reports)
    compute_subtype_difficulty(reports)

    print("\n" + "=" * 60)
    print("Statistics computation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
