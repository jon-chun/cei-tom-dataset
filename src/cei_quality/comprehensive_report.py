"""
Comprehensive QA Report Generator.

Generates detailed markdown reports with rankings by annotator, subtype,
and individual scenario. Integrates with the CEI Quality Pipeline.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from cei_quality.pipeline import CEIQualityPipeline


def generate_comprehensive_report(
    pipeline: "CEIQualityPipeline",
    output_path: Path,
) -> str:
    """
    Generate comprehensive QA markdown report.

    Args:
        pipeline: The CEIQualityPipeline instance with computed results
        output_path: Path to write the report

    Returns:
        The generated report content as a string
    """
    lines = [
        "# CEI Quality Pipeline - Comprehensive QA Report",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        f"**Run ID:** {pipeline.run_metadata.run_id if pipeline.run_metadata else 'N/A'}",
        f"**Config Hash:** {pipeline.run_metadata.config_hash if pipeline.run_metadata else 'N/A'}",
        "",
        "---",
        "",
        "## Table of Contents",
        "",
        "1. [Executive Summary](#executive-summary)",
        "2. [Quality by Annotator](#quality-by-annotator-15-total)",
        "3. [Quality by Scenario Subtype](#quality-by-scenario-subtype-5-total)",
        "4. [Quality by Individual Scenario](#quality-by-individual-scenario-300-total)",
        "5. [Prioritized Review List](#prioritized-review-list)",
        "6. [Systematic Outlier Annotators](#systematic-outlier-annotators)",
        "7. [Confusion Matrix Analysis](#confusion-matrix-analysis)",
        "8. [Agreement Metrics](#agreement-metrics)",
        "9. [Methodology](#methodology)",
        "",
        "---",
        "",
    ]

    # === EXECUTIVE SUMMARY ===
    lines.extend(_generate_executive_summary(pipeline))

    # === QUALITY BY ANNOTATOR ===
    lines.extend(_generate_annotator_rankings(pipeline))

    # === QUALITY BY SUBTYPE ===
    lines.extend(_generate_subtype_rankings(pipeline))

    # === QUALITY BY SCENARIO ===
    lines.extend(_generate_scenario_rankings(pipeline))

    # === PRIORITIZED REVIEW LIST ===
    lines.extend(_generate_prioritized_review_list(pipeline))

    # === SYSTEMATIC OUTLIERS ===
    lines.extend(_generate_systematic_outliers_section(pipeline))

    # === CONFUSION MATRIX ===
    lines.extend(_generate_confusion_matrix_section(pipeline))

    # === AGREEMENT METRICS ===
    lines.extend(_generate_agreement_metrics_section(pipeline))

    # === METHODOLOGY ===
    lines.extend(_generate_methodology_section())

    report_content = "\n".join(lines)

    # Write report
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(report_content)

    return report_content


def _generate_executive_summary(pipeline: "CEIQualityPipeline") -> List[str]:
    """Generate executive summary section."""
    lines = [
        "## Executive Summary",
        "",
    ]

    total_scenarios = len(pipeline.quality_reports)
    total_annotators = len(pipeline.annotator_reports)
    total_subtypes = len(pipeline.fleiss_kappa_by_subtype)

    # Calculate overall statistics
    quality_scores = [r.quality_score for r in pipeline.quality_reports]
    avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0

    flagged = sum(1 for r in pipeline.quality_reports if r.needs_review)
    critical_issues = sum(r.critical_count for r in pipeline.quality_reports)
    major_issues = sum(r.major_count for r in pipeline.quality_reports)

    overall_kappa = (
        sum(pipeline.fleiss_kappa_by_subtype.values()) / total_subtypes
        if total_subtypes > 0
        else 0
    )

    # Quality distribution
    excellent = sum(1 for r in pipeline.quality_reports if r.quality_score >= 0.95)
    good = sum(1 for r in pipeline.quality_reports if 0.85 <= r.quality_score < 0.95)
    acceptable = sum(
        1 for r in pipeline.quality_reports if 0.70 <= r.quality_score < 0.85
    )
    questionable = sum(
        1 for r in pipeline.quality_reports if 0.50 <= r.quality_score < 0.70
    )
    poor = sum(1 for r in pipeline.quality_reports if r.quality_score < 0.50)

    lines.extend(
        [
            "### Overview",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Total Scenarios | {total_scenarios} |",
            f"| Total Annotators | {total_annotators} |",
            f"| Scenario Subtypes | {total_subtypes} |",
            f"| **Average Quality Score** | **{avg_quality:.3f}** |",
            f"| Overall Fleiss' κ | {overall_kappa:.3f} |",
            f"| Scenarios Flagged for Review | {flagged} ({flagged/total_scenarios*100:.1f}%) |",
            f"| Critical Issues | {critical_issues} |",
            f"| Major Issues | {major_issues} |",
            "",
            "### Quality Distribution",
            "",
            "| Quality Bucket | Count | Percentage |",
            "|----------------|-------|------------|",
            f"| Excellent (≥0.95) | {excellent} | {excellent/total_scenarios*100:.1f}% |",
            f"| Good (0.85-0.95) | {good} | {good/total_scenarios*100:.1f}% |",
            f"| Acceptable (0.70-0.85) | {acceptable} | {acceptable/total_scenarios*100:.1f}% |",
            f"| Questionable (0.50-0.70) | {questionable} | {questionable/total_scenarios*100:.1f}% |",
            f"| Poor (<0.50) | {poor} | {poor/total_scenarios*100:.1f}% |",
            "",
            "### Interpretation Guide",
            "",
            "**Quality Score Thresholds:**",
            "- **Excellent (≥0.95)**: No action needed",
            "- **Good (0.85-0.95)**: Minor review if time permits",
            "- **Acceptable (0.70-0.85)**: Review recommended, likely minor corrections",
            "- **Questionable (0.50-0.70)**: **Mandatory review** - likely annotation errors",
            "- **Poor (<0.50)**: **Priority review** - significant annotation problems",
            "",
            "**Fleiss' κ Interpretation:**",
            "- κ ≥ 0.81: Almost perfect agreement",
            "- κ 0.61-0.80: Substantial agreement",
            "- κ 0.41-0.60: Moderate agreement (acceptable for pragmatic ambiguity)",
            "- κ 0.21-0.40: Fair agreement (may indicate task clarity issues)",
            "- κ < 0.21: Poor agreement (requires investigation)",
            "",
            "---",
            "",
        ]
    )

    return lines


def _generate_annotator_rankings(pipeline: "CEIQualityPipeline") -> List[str]:
    """Generate annotator rankings section."""
    lines = [
        "## Quality by Annotator (15 Total)",
        "",
        "Annotators ranked by quality issues (descending order of problems):",
        "",
    ]

    # Create annotator summaries from file reports
    annotator_stats: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {
            "subtypes": [],
            "total_annotations": 0,
            "total_issues": 0,
            "critical_issues": 0,
            "major_issues": 0,
            "mean_quality": 0.0,
            "mean_lead_time": 0.0,
            "fast_count": 0,
            "needs_review": False,
            "review_reasons": [],
            "quality_scores": [],
        }
    )

    for fr in pipeline.file_reports:
        stats = annotator_stats[fr.annotator_name]
        stats["subtypes"].append(fr.subtype)
        stats["total_annotations"] += fr.record_count
        stats["total_issues"] += fr.issue_count
        stats["quality_scores"].append(fr.mean_quality_score)
        stats["mean_lead_time"] += fr.mean_lead_time
        stats["fast_count"] += fr.suspicious_fast_count
        if fr.needs_full_review:
            stats["needs_review"] = True
            stats["review_reasons"].append(f"[{fr.subtype}] {fr.review_reason}")

    # Calculate averages and count critical/major from quality reports
    for annotator in annotator_stats:
        scores = annotator_stats[annotator]["quality_scores"]
        if scores:
            annotator_stats[annotator]["mean_quality"] = sum(scores) / len(scores)
        subtypes = annotator_stats[annotator]["subtypes"]
        if subtypes:
            annotator_stats[annotator]["mean_lead_time"] /= len(subtypes)

    # Rank by total issues (descending)
    ranked = sorted(
        annotator_stats.items(),
        key=lambda x: (-x[1]["total_issues"], x[1]["mean_quality"]),
    )

    # Generate table
    lines.extend(
        [
            "| Rank | Annotator | Subtypes | Annotations | Issues | Quality | Avg Lead Time | Fast Ann. | Needs Review |",
            "|------|-----------|----------|-------------|--------|---------|---------------|-----------|--------------|",
        ]
    )

    for rank, (name, stats) in enumerate(ranked, 1):
        subtypes_str = ", ".join(sorted(set(stats["subtypes"])))[:30]
        review_icon = "⚠️" if stats["needs_review"] else "✅"
        lines.append(
            f"| {rank} | **{name}** | {subtypes_str} | {stats['total_annotations']} | "
            f"{stats['total_issues']} | {stats['mean_quality']:.3f} | "
            f"{stats['mean_lead_time']:.1f}s | {stats['fast_count']} | {review_icon} |"
        )

    lines.append("")

    # Detailed issues for flagged annotators
    flagged_annotators = [name for name, stats in ranked if stats["needs_review"]]
    if flagged_annotators:
        lines.extend(
            [
                "### Annotators Requiring Review",
                "",
            ]
        )
        for name in flagged_annotators:
            stats = annotator_stats[name]
            lines.extend(
                [
                    f"#### {name}",
                    "",
                ]
            )
            for reason in stats["review_reasons"]:
                lines.append(f"- {reason}")
            lines.append("")

    lines.extend([
        "### How to Investigate Annotator Issues",
        "",
        "1. **High issue count with low quality**: Check for systematic errors",
        "2. **Many fast annotations**: Review for rushing (< 15s suggests insufficient reading)",
        "3. **⚠️ Needs Review flag**: Indicates straight-lining or other intra-annotator issues",
        "",
        "**Resolution Steps:**",
        "- Compare annotator's labels against the majority for disagreement patterns",
        "- Look for emotion confusion pairs (e.g., always labeling fear as surprise)",
        "- Check if VAD values are consistent with emotion labels",
        "- Consider retraining if patterns are systematic",
        "",
        "---",
        "",
    ])
    return lines


def _generate_subtype_rankings(pipeline: "CEIQualityPipeline") -> List[str]:
    """Generate subtype rankings section."""
    lines = [
        "## Quality by Scenario Subtype (5 Total)",
        "",
        "Subtypes ranked by quality issues (descending order of problems):",
        "",
    ]

    # Aggregate by subtype
    subtype_stats: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {
            "scenario_count": 0,
            "total_issues": 0,
            "critical_count": 0,
            "major_count": 0,
            "flagged_count": 0,
            "quality_scores": [],
            "fleiss_kappa": 0.0,
        }
    )

    for r in pipeline.quality_reports:
        stats = subtype_stats[r.subtype]
        stats["scenario_count"] += 1
        stats["total_issues"] += r.issue_count
        stats["critical_count"] += r.critical_count
        stats["major_count"] += r.major_count
        stats["quality_scores"].append(r.quality_score)
        if r.needs_review:
            stats["flagged_count"] += 1

    # Add kappa values
    for subtype, kappa in pipeline.fleiss_kappa_by_subtype.items():
        subtype_stats[subtype]["fleiss_kappa"] = kappa

    # Calculate means
    for subtype in subtype_stats:
        scores = subtype_stats[subtype]["quality_scores"]
        if scores:
            subtype_stats[subtype]["mean_quality"] = sum(scores) / len(scores)
            subtype_stats[subtype]["min_quality"] = min(scores)
        else:
            subtype_stats[subtype]["mean_quality"] = 0
            subtype_stats[subtype]["min_quality"] = 0

    # Rank by total issues
    ranked = sorted(
        subtype_stats.items(),
        key=lambda x: (-x[1]["total_issues"], x[1]["mean_quality"]),
    )

    lines.extend(
        [
            "| Rank | Subtype | Scenarios | Issues | Critical | Major | Flagged | Mean Quality | Min Quality | Fleiss' κ |",
            "|------|---------|-----------|--------|----------|-------|---------|--------------|-------------|-----------|",
        ]
    )

    for rank, (subtype, stats) in enumerate(ranked, 1):
        lines.append(
            f"| {rank} | **{subtype}** | {stats['scenario_count']} | {stats['total_issues']} | "
            f"{stats['critical_count']} | {stats['major_count']} | {stats['flagged_count']} | "
            f"{stats['mean_quality']:.3f} | {stats['min_quality']:.3f} | {stats['fleiss_kappa']:.3f} |"
        )

    lines.extend(
        [
            "",
            "### Issue Breakdown by Subtype",
            "",
        ]
    )

    # Issue types by subtype
    for subtype, _ in ranked:
        subtype_issues: Dict[str, int] = defaultdict(int)
        for r in pipeline.quality_reports:
            if r.subtype == subtype:
                for issue in r.issues:
                    subtype_issues[issue.flag.value] += 1

        if subtype_issues:
            lines.extend(
                [
                    f"#### {subtype}",
                    "",
                    "| Issue Type | Count |",
                    "|------------|-------|",
                ]
            )
            for flag, count in sorted(subtype_issues.items(), key=lambda x: -x[1])[:10]:
                lines.append(f"| {flag} | {count} |")
            lines.append("")

    lines.extend([
        "### Subtype-Specific Guidance",
        "",
        "**Common issues by subtype:**",
        "- **passive-aggression**: Often confused with sadness/anger; check utterance tone",
        "- **sarcasm-irony**: High disagreement expected; focus on context clues",
        "- **mixed-signals**: VAD inconsistency common; verify listener perspective",
        "- **deflection-misdirection**: May be confused with fear/anticipation",
        "- **strategic-politeness**: Trust often misidentified; review social context",
        "",
        "**When a subtype has low κ:**",
        "1. Review task instructions for that subtype",
        "2. Check if scenarios are genuinely ambiguous (expected for pragmatics)",
        "3. Identify specific emotion pairs causing confusion",
        "4. Consider providing additional annotator training",
        "",
        "---",
        "",
    ])
    return lines


def _generate_scenario_rankings(pipeline: "CEIQualityPipeline") -> List[str]:
    """Generate individual scenario rankings section."""
    lines = [
        "## Quality by Individual Scenario (300 Total)",
        "",
        "Top 50 scenarios with most issues (descending order of problems):",
        "",
    ]

    # Rank scenarios by issue count then quality score
    ranked = sorted(
        pipeline.quality_reports,
        key=lambda x: (-x.issue_count, -x.critical_count, x.quality_score),
    )

    lines.extend(
        [
            "| Rank | ID | Subtype | Quality | Issues | Critical | Major | Priority | Review Reasons |",
            "|------|----|---------|---------|--------|----------|-------|----------|----------------|",
        ]
    )

    for rank, r in enumerate(ranked[:50], 1):
        reasons = "; ".join(r.review_reasons[:2]) if r.review_reasons else "-"
        if len(reasons) > 50:
            reasons = reasons[:47] + "..."
        lines.append(
            f"| {rank} | {r.scenario_id} | {r.subtype[:15]} | {r.quality_score:.3f} | "
            f"{r.issue_count} | {r.critical_count} | {r.major_count} | {r.review_priority} | {reasons} |"
        )

    lines.extend(
        [
            "",
            "### Quality Score Distribution by Scenario",
            "",
        ]
    )

    # Create histogram-like view
    buckets = {
        "0.00-0.50": 0,
        "0.50-0.60": 0,
        "0.60-0.70": 0,
        "0.70-0.80": 0,
        "0.80-0.90": 0,
        "0.90-0.95": 0,
        "0.95-1.00": 0,
    }

    for r in pipeline.quality_reports:
        q = r.quality_score
        if q < 0.50:
            buckets["0.00-0.50"] += 1
        elif q < 0.60:
            buckets["0.50-0.60"] += 1
        elif q < 0.70:
            buckets["0.60-0.70"] += 1
        elif q < 0.80:
            buckets["0.70-0.80"] += 1
        elif q < 0.90:
            buckets["0.80-0.90"] += 1
        elif q < 0.95:
            buckets["0.90-0.95"] += 1
        else:
            buckets["0.95-1.00"] += 1

    lines.extend(
        [
            "| Score Range | Count | Bar |",
            "|-------------|-------|-----|",
        ]
    )

    max_count = max(buckets.values()) if buckets.values() else 1
    for range_str, count in buckets.items():
        bar_len = int((count / max_count) * 30) if max_count > 0 else 0
        bar = "█" * bar_len
        lines.append(f"| {range_str} | {count} | {bar} |")

    lines.extend([
        "",
        "### Scenario Review Workflow",
        "",
        "**For each flagged scenario:**",
        "1. Open `reports/report_qa_records.md` to see all 3 annotator labels",
        "2. Read the scenario context (situation + utterance)",
        "3. Remember: labels describe the **LISTENER's** emotional response",
        "4. Identify which annotator(s) are incorrect",
        "5. Edit the source JSON file to correct the label",
        "",
        "**Common scenario issues:**",
        "- **No majority emotion**: All 3 annotators disagree - choose most contextually appropriate",
        "- **High VAD variance**: Check if VAD values match the emotion (see Emotion-VAD profiles in manual)",
        "- **LLM flagged**: Semantic check failed - verify emotion makes sense for context",
        "",
        "---",
        "",
    ])
    return lines


def _generate_prioritized_review_list(pipeline: "CEIQualityPipeline") -> List[str]:
    """Generate combined prioritized review list."""
    lines = [
        "## Prioritized Review List",
        "",
        "Combined ranking of items requiring human review, synthesizing annotator,",
        "subtype, and scenario-level issues. Sorted by priority (descending).",
        "",
    ]

    # Collect all items needing review
    review_items: List[Dict[str, Any]] = []

    # Add scenario-level items
    for r in pipeline.quality_reports:
        if r.needs_review or r.quality_score < 0.70:
            review_items.append(
                {
                    "type": "scenario",
                    "id": f"scenario_{r.subtype}_{r.scenario_id}",
                    "subtype": r.subtype,
                    "scenario_id": r.scenario_id,
                    "priority": r.review_priority,
                    "quality_score": r.quality_score,
                    "issues": r.issue_count,
                    "critical": r.critical_count,
                    "reasons": r.review_reasons,
                    "description": f"Scenario {r.scenario_id} ({r.subtype})",
                }
            )

    # Add annotator-level items
    for fr in pipeline.file_reports:
        if fr.needs_full_review:
            review_items.append(
                {
                    "type": "annotator",
                    "id": f"annotator_{fr.annotator_name}_{fr.subtype}",
                    "subtype": fr.subtype,
                    "scenario_id": None,
                    "priority": 10,  # High priority for annotator issues
                    "quality_score": fr.mean_quality_score,
                    "issues": fr.issue_count,
                    "critical": 0,
                    "reasons": [fr.review_reason],
                    "description": f"Annotator {fr.annotator_name} ({fr.subtype})",
                }
            )

    # Sort by priority desc, then quality score asc
    review_items.sort(key=lambda x: (-x["priority"], x["quality_score"]))

    lines.extend(
        [
            f"**Total items requiring review:** {len(review_items)}",
            "",
            "| Rank | Type | Description | Priority | Quality | Issues | Reasons |",
            "|------|------|-------------|----------|---------|--------|---------|",
        ]
    )

    for rank, item in enumerate(review_items[:100], 1):
        reasons = "; ".join(item["reasons"][:2]) if item["reasons"] else "-"
        if len(reasons) > 40:
            reasons = reasons[:37] + "..."
        type_icon = "📋" if item["type"] == "scenario" else "👤"
        lines.append(
            f"| {rank} | {type_icon} {item['type']} | {item['description'][:30]} | "
            f"{item['priority']} | {item['quality_score']:.3f} | {item['issues']} | {reasons} |"
        )

    if len(review_items) > 100:
        lines.append(
            f"| ... | ... | *({len(review_items) - 100} more items)* | ... | ... | ... | ... |"
        )

    lines.extend(
        [
            "",
            "### Review Priority Explanation",
            "",
            "| Priority | Criteria |",
            "|----------|----------|",
            "| 10 | Critical issues or annotator-level problems |",
            "| 9 | No majority emotion agreement |",
            "| 8 | Quality score < 0.50 |",
            "| 7 | Multiple major issues |",
            "| 6 | LLM flagged implausibility |",
            "| 1-5 | Lower priority issues |",
            "",
            "---",
            "",
        ]
    )

    return lines


def _generate_systematic_outliers_section(pipeline: "CEIQualityPipeline") -> List[str]:
    """Generate systematic outlier annotators section."""
    lines = [
        "## Systematic Outlier Annotators",
        "",
        "Annotators who systematically disagree with the majority in 2-1 splits.",
        "These may indicate calibration issues requiring retraining or review.",
        "",
    ]

    if not pipeline.systematic_outliers_by_subtype:
        lines.append("*No systematic outlier data available.*")
        lines.extend(["", "---", ""])
        return lines

    # Collect all outliers across subtypes
    all_outliers: Dict[str, Dict[str, Any]] = {}

    for subtype, outlier_data in pipeline.systematic_outliers_by_subtype.items():
        annotator_stats = outlier_data.get("annotator_stats", {})

        for ann_name, stats in annotator_stats.items():
            if ann_name not in all_outliers:
                all_outliers[ann_name] = {
                    "subtypes": [],
                    "avg_agreement": [],
                    "minority_rate": [],
                    "is_outlier": False,
                    "patterns": [],
                }

            all_outliers[ann_name]["subtypes"].append(subtype)
            all_outliers[ann_name]["avg_agreement"].append(stats["avg_pairwise_agreement"])
            all_outliers[ann_name]["minority_rate"].append(stats["minority_rate"])

            if stats.get("is_outlier"):
                all_outliers[ann_name]["is_outlier"] = True

            # Get minority patterns
            patterns = outlier_data.get("minority_patterns", {}).get(ann_name, [])
            for p in patterns:
                all_outliers[ann_name]["patterns"].append({
                    **p,
                    "subtype": subtype,
                })

    # Summary table
    lines.extend([
        "### Annotator Agreement Summary",
        "",
        "| Annotator | Subtype | Avg Agreement | Times Minority | Minority Rate | Outlier? |",
        "|-----------|---------|---------------|----------------|---------------|----------|",
    ])

    for ann_name, data in sorted(all_outliers.items(), key=lambda x: min(x[1]["avg_agreement"])):
        for i, subtype in enumerate(data["subtypes"]):
            outlier_icon = "⚠️" if data["is_outlier"] else "✅"
            lines.append(
                f"| {ann_name} | {subtype} | {data['avg_agreement'][i]:.1%} | "
                f"— | {data['minority_rate'][i]:.1%} | {outlier_icon} |"
            )

    lines.append("")

    # Detailed patterns for flagged annotators
    flagged = [name for name, data in all_outliers.items() if data["is_outlier"]]
    if flagged:
        lines.extend([
            "### Minority Opinion Patterns",
            "",
            "Scenarios where flagged annotators disagreed with the majority:",
            "",
        ])

        for ann_name in flagged:
            patterns = all_outliers[ann_name]["patterns"][:5]  # Top 5
            if patterns:
                lines.extend([
                    f"#### {ann_name}",
                    "",
                    "| Subtype | Scenario | Their Label | Majority Label |",
                    "|---------|----------|-------------|----------------|",
                ])
                for p in patterns:
                    lines.append(
                        f"| {p['subtype']} | {p['scenario_id']} | "
                        f"{p['annotator_emotion']} | {p['majority_emotion']} |"
                    )
                lines.append("")

    lines.extend([
        "### How to Address Systematic Outliers",
        "",
        "1. **Review calibration**: Check if annotator understood the task correctly",
        "2. **Examine patterns**: Look for consistent confusion between emotion pairs",
        "3. **Consider context**: Some subtypes may genuinely have higher ambiguity",
        "4. **Retrain if needed**: Provide additional training or clarification",
        "",
        "---",
        "",
    ])

    return lines


def _generate_confusion_matrix_section(pipeline: "CEIQualityPipeline") -> List[str]:
    """Generate confusion matrix analysis section."""
    lines = [
        "## Confusion Matrix Analysis",
        "",
        "Emotion confusion patterns across all annotator pairs:",
        "",
    ]

    if not pipeline.confusion_matrices_by_subtype:
        lines.append("*No confusion matrix data available.*")
        lines.extend(["", "---", ""])
        return lines

    # Aggregate all confusions
    all_confusions: Dict[Tuple[str, str], int] = defaultdict(int)

    for subtype, matrix_data in pipeline.confusion_matrices_by_subtype.items():
        for conf in matrix_data.get("common_confusions", []):
            key = tuple(sorted([conf["emotion_1"], conf["emotion_2"]]))
            all_confusions[key] += conf["count"]

    # Sort by frequency
    sorted_confusions = sorted(all_confusions.items(), key=lambda x: -x[1])

    lines.extend(
        [
            "### Most Common Emotion Confusions",
            "",
            "| Rank | Emotion 1 | Emotion 2 | Count |",
            "|------|-----------|-----------|-------|",
        ]
    )

    for rank, ((e1, e2), count) in enumerate(sorted_confusions[:15], 1):
        lines.append(f"| {rank} | {e1} | {e2} | {count} |")

    lines.extend(
        [
            "",
            "### Confusion by Subtype",
            "",
        ]
    )

    for subtype, matrix_data in pipeline.confusion_matrices_by_subtype.items():
        agreement_rate = matrix_data.get("agreement_rate", 0)
        lines.extend(
            [
                f"#### {subtype}",
                f"- Agreement rate: {agreement_rate:.1%}",
                "- Top confusions:",
            ]
        )
        for conf in matrix_data.get("common_confusions", [])[:5]:
            lines.append(
                f"  - {conf['emotion_1']} ↔ {conf['emotion_2']}: {conf['count']}"
            )
        lines.append("")

    lines.extend([
        "### Interpreting Confusion Patterns",
        "",
        "**Expected confusions (semantically related):**",
        "- sadness ↔ fear (both low-valence, controlled)",
        "- anger ↔ disgust (both high-arousal, negative)",
        "- joy ↔ anticipation (both positive, high-energy)",
        "- trust ↔ anticipation (both approach-oriented)",
        "",
        "**Problematic confusions (opposite emotions):**",
        "- joy ↔ sadness (valence mismatch)",
        "- trust ↔ disgust (opposite on Plutchik wheel)",
        "- fear ↔ anger (different dominance profiles)",
        "",
        "**Action items:**",
        "1. High-frequency expected confusions: Add clarifying examples to guidelines",
        "2. Opposite-emotion confusions: Investigate if scenarios are ambiguous",
        "3. Annotator-specific patterns: Check in Systematic Outliers section",
        "",
        "---",
        "",
    ])
    return lines


def _generate_agreement_metrics_section(pipeline: "CEIQualityPipeline") -> List[str]:
    """Generate detailed agreement metrics section."""
    from cei_quality.validators.agreement import InterAnnotatorValidator

    lines = [
        "## Agreement Metrics",
        "",
        "### Fleiss' Kappa with Confidence Intervals",
        "",
        "| Subtype | κ | 95% CI Lower | 95% CI Upper | SE | Interpretation |",
        "|---------|---|--------------|--------------|-------|----------------|",
    ]

    for subtype, kappa_data in pipeline.kappa_with_ci_by_subtype.items():
        kappa = kappa_data.get("kappa", 0)
        ci_lower = kappa_data.get("ci_lower", 0)
        ci_upper = kappa_data.get("ci_upper", 0)
        se = kappa_data.get("se", 0)
        interp = InterAnnotatorValidator.interpret_kappa(kappa)
        lines.append(
            f"| {subtype} | {kappa:.3f} | {ci_lower:.3f} | {ci_upper:.3f} | {se:.3f} | {interp} |"
        )

    lines.extend(
        [
            "",
            "### Weighted Kappa for VAD Dimensions",
            "",
            "| Subtype | Valence | Arousal | Dominance |",
            "|---------|---------|---------|-----------|",
        ]
    )

    for subtype, vad_kappas in pipeline.weighted_kappa_by_subtype.items():
        v = vad_kappas.get("valence", 0)
        a = vad_kappas.get("arousal", 0)
        d = vad_kappas.get("dominance", 0)
        lines.append(f"| {subtype} | {v:.3f} | {a:.3f} | {d:.3f} |")

    lines.extend([
        "",
        "### VAD Dimension Interpretation",
        "",
        "**Valence (pleasant ↔ unpleasant):**",
        "- High κ (>0.7): Strong agreement on emotion positivity/negativity",
        "- Low κ (<0.4): Annotators disagree on whether situations are positive or negative",
        "",
        "**Arousal (calm ↔ aroused):**",
        "- High κ (>0.7): Agreement on intensity of emotional response",
        "- Low κ (<0.4): Disagreement on whether responses are intense or subdued",
        "",
        "**Dominance (controlled ↔ dominant):**",
        "- High κ (>0.7): Agreement on listener's sense of control",
        "- Low κ (<0.4): Disagreement on power dynamics in scenarios",
        "",
        "---",
        "",
    ])
    return lines


def _generate_methodology_section() -> List[str]:
    """Generate methodology explanation section."""
    return [
        "## Methodology",
        "",
        "### Quality Score Formula",
        "",
        "```",
        "quality_score = 0.25 × schema + 0.20 × consistency + 0.35 × agreement + 0.20 × plausibility",
        "```",
        "",
        "### Validation Stages",
        "",
        "1. **Stage 1A (Schema)**: JSON structure and required fields",
        "2. **Stage 1B (Consistency)**: Duplicates, missing IDs, invalid labels, lead time anomalies",
        "3. **Stage 1C (Agreement)**: Inter-annotator agreement (Fleiss' κ), emotion/VAD variance",
        "4. **Stage 1D (Plausibility)**: VAD-emotion consistency, LLM semantic validation",
        "",
        "### Issue Severity Weights",
        "",
        "| Severity | Weight |",
        "|----------|--------|",
        "| Critical | 0.40 |",
        "| Major | 0.20 |",
        "| Minor | 0.05 |",
        "| Info | 0.00 |",
        "",
        "### Intra-Annotator Checks",
        "",
        "- **Straight-lining**: Flag if >80% same emotion used",
        "- **MAD Dwell Outliers**: Modified Z-score with 2.5 MAD threshold",
        "- **Self-contradictions**: Same emotion with opposite valences",
        "",
        "---",
        "",
        "*Report generated by CEI Quality Pipeline v1.0.0*",
    ]
