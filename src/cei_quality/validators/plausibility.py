"""
Stage 1D: Plausibility Validation

Validates that annotations are semantically plausible given the scenarios.

Checks performed:
- VAD-emotion consistency (deterministic)
- Listener emotion plausibility (optional LLM check)
- Speaker/listener perspective confusion detection
- Corroborating evidence validation for LLM judgments

IMPORTANT: Annotations describe the LISTENER's emotional response,
not the speaker's emotion. This is a common source of confusion.

LLM CORROBORATION: LLM judgments are only counted if they have
supporting deterministic evidence (VAD mismatch, low agreement, fast lead time).
This prevents over-reliance on potentially unreliable LLM confidence scores.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any, Dict, List, Optional, Set, TYPE_CHECKING

from cei_quality.models import (
    AnnotationData,
    PragmaticSubtype,
    QualityFlag,
    QualityIssue,
    ScenarioData,
)

if TYPE_CHECKING:
    from cei_quality.config import CEIConfig

logger = logging.getLogger(__name__)


# Flags that can corroborate LLM judgments
CORROBORATING_FLAGS: Set[QualityFlag] = {
    QualityFlag.VAD_EMOTION_MISMATCH,
    QualityFlag.HIGH_VAD_DISAGREEMENT,
    QualityFlag.HIGH_EMOTION_DISAGREEMENT,
    QualityFlag.NO_MAJORITY_EMOTION,
    QualityFlag.SUSPICIOUS_LEAD_TIME,
    QualityFlag.DWELL_TIME_OUTLIER,
}


class PlausibilityChecker:
    """
    Stage 1D: Check plausibility of annotations.

    This validator ensures that annotations make sense given the
    scenario content. It includes both deterministic checks
    (VAD-emotion consistency) and optional LLM-based verification.

    Key insight: Annotations describe the LISTENER's emotional
    response to hearing the utterance, not the speaker's emotion.

    Example:
        >>> checker = PlausibilityChecker(config)
        >>> issues = checker.check_scenario(scenario, annotations, subtype)
        >>> # With LLM (optional):
        >>> issues = checker.check_with_llm([(scenario, annotations, subtype)])
    """

    # System prompt for LLM plausibility checking
    LLM_SYSTEM_PROMPT = """You are a quality assurance expert reviewing emotion annotations for a psychology research dataset.

CRITICAL CONTEXT: These annotations describe the LISTENER's emotional response to hearing the utterance, NOT the speaker's emotion. This is a stimulus-response paradigm:
- Stimulus: A scenario with situation, speaker role, listener role, and utterance
- Response: How the LISTENER would feel after hearing this utterance

Your task is to identify potentially incorrect or suspicious annotations.

For each scenario, evaluate:
1. LISTENER_EMOTION_PLAUSIBILITY: Does this emotion make sense as the LISTENER's response?
   - Consider: How would the listener feel hearing this? Not how the speaker feels saying it.
   
2. VAD_CONSISTENCY: Are the Valence/Arousal/Dominance ratings consistent with the labeled emotion?
   - Typical profiles:
     * Joy: positive valence, moderate-high arousal, moderate-high dominance
     * Sadness: negative valence, low arousal, low dominance
     * Anger: negative valence, high arousal, high dominance
     * Fear: negative valence, high arousal, LOW dominance
     * Trust: positive valence, low arousal, moderate dominance
     * Disgust: negative valence, moderate arousal, moderate dominance
     * Surprise: varies by valence, high arousal, low-moderate dominance
     * Anticipation: varies, moderate arousal, moderate dominance

3. PERSPECTIVE_CHECK: Is there any sign the annotator confused speaker vs listener perspective?

Be CONSERVATIVE - only flag clear issues. Annotators may have valid interpretations you didn't consider."""

    def __init__(self, config: CEIConfig) -> None:
        """
        Initialize plausibility checker.

        Args:
            config: CEI configuration instance
        """
        self.config = config
        self._llm_client = None

    def check_scenario(
        self,
        scenario: ScenarioData,
        annotations: List[AnnotationData],
        subtype: PragmaticSubtype,
    ) -> List[QualityIssue]:
        """
        Check plausibility of annotations (deterministic checks only).

        Args:
            scenario: The scenario data
            annotations: All annotations for this scenario
            subtype: The pragmatic subtype

        Returns:
            List of plausibility issues
        """
        issues: List[QualityIssue] = []

        for ann in annotations:
            # Check VAD-emotion consistency
            vad_issues = self._check_vad_emotion_consistency(
                scenario.scenario_id,
                ann,
            )
            issues.extend(vad_issues)

        return issues

    def _check_vad_emotion_consistency(
        self,
        scenario_id: int,
        ann: AnnotationData,
    ) -> List[QualityIssue]:
        """
        Check if VAD ratings match expected profile for the labeled emotion.

        Uses configurable tolerance to allow for natural variation.
        """
        issues: List[QualityIssue] = []

        emotion = ann.plutchik_emotion.lower().strip()
        profile = self.config.get_emotion_profile(emotion)

        if profile is None:
            # Unknown emotion, can't check consistency
            return issues

        vad = ann.get_vad_numeric(self.config)
        tolerance = self.config.quality.vad_tolerance

        # Check valence
        v = vad["v"]
        v_range = profile.valence
        if v < v_range[0] - tolerance or v > v_range[1] + tolerance:
            issues.append(
                QualityIssue(
                    flag=QualityFlag.VAD_EMOTION_MISMATCH,
                    severity="minor",
                    message=f"Valence ({v:.1f}) unusual for '{emotion}' (expected {v_range})",
                    details={
                        "scenario_id": scenario_id,
                        "annotator": ann.annotator_name,
                        "emotion": emotion,
                        "dimension": "valence",
                        "value": v,
                        "expected_range": list(v_range),
                        "tolerance": tolerance,
                    },
                )
            )

        # Check arousal
        a = vad["a"]
        a_range = profile.arousal
        if a < a_range[0] - tolerance or a > a_range[1] + tolerance:
            issues.append(
                QualityIssue(
                    flag=QualityFlag.VAD_EMOTION_MISMATCH,
                    severity="minor",
                    message=f"Arousal ({a:.1f}) unusual for '{emotion}' (expected {a_range})",
                    details={
                        "scenario_id": scenario_id,
                        "annotator": ann.annotator_name,
                        "emotion": emotion,
                        "dimension": "arousal",
                        "value": a,
                        "expected_range": list(a_range),
                        "tolerance": tolerance,
                    },
                )
            )

        # Check dominance
        d = vad["d"]
        d_range = profile.dominance
        if d < d_range[0] - tolerance or d > d_range[1] + tolerance:
            issues.append(
                QualityIssue(
                    flag=QualityFlag.VAD_EMOTION_MISMATCH,
                    severity="minor",
                    message=f"Dominance ({d:.1f}) unusual for '{emotion}' (expected {d_range})",
                    details={
                        "scenario_id": scenario_id,
                        "annotator": ann.annotator_name,
                        "emotion": emotion,
                        "dimension": "dominance",
                        "value": d,
                        "expected_range": list(d_range),
                        "tolerance": tolerance,
                    },
                )
            )

        return issues

    def check_with_llm(
        self,
        scenarios: List[tuple[ScenarioData, List[AnnotationData], PragmaticSubtype]],
        batch_size: int = 5,
    ) -> Dict[int, List[QualityIssue]]:
        """
        Check plausibility using LLM (requires API key).

        This is an expensive operation - use sparingly for flagged items.

        Args:
            scenarios: List of (scenario, annotations, subtype) tuples
            batch_size: Number of scenarios per API call

        Returns:
            Dict mapping scenario_id -> list of LLM-flagged issues
        """
        if not self.config.llm.enabled:
            logger.warning("LLM checks disabled in config")
            return {}

        issues_by_scenario: Dict[int, List[QualityIssue]] = {}

        try:
            client = self._get_llm_client()
        except Exception as e:
            logger.error(f"Failed to initialize LLM client: {e}")
            return {}

        # Process in batches
        for i in range(0, len(scenarios), batch_size):
            batch = scenarios[i : i + batch_size]

            try:
                batch_issues = self._check_batch_with_llm(client, batch)
                issues_by_scenario.update(batch_issues)
            except Exception as e:
                logger.error(f"LLM batch check failed: {e}")

        return issues_by_scenario

    def _get_llm_client(self) -> Any:
        """Get or create LLM client."""
        if self._llm_client is not None:
            return self._llm_client

        provider = self.config.llm.provider

        if provider == "openai":
            import os
            from openai import OpenAI

            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set")
            self._llm_client = OpenAI(api_key=api_key)
        elif provider == "anthropic":
            import os
            from anthropic import Anthropic

            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY environment variable not set")
            self._llm_client = Anthropic(api_key=api_key)
        else:
            raise ValueError(f"Unknown LLM provider: {provider}")

        return self._llm_client

    def _check_batch_with_llm(
        self,
        client: Any,
        batch: List[tuple[ScenarioData, List[AnnotationData], PragmaticSubtype]],
    ) -> Dict[int, List[QualityIssue]]:
        """Process a batch of scenarios with LLM."""
        import json
        from collections import Counter

        issues_by_scenario: Dict[int, List[QualityIssue]] = {}

        # Build prompt
        prompt = self._build_batch_prompt(batch)

        provider = self.config.llm.provider
        model = self.config.llm.model
        temperature = self.config.llm.temperature

        if provider == "openai":
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": self.LLM_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                response_format={"type": "json_object"},
            )
            result_text = response.choices[0].message.content
        elif provider == "anthropic":
            response = client.messages.create(
                model=model,
                max_tokens=4096,
                system=self.LLM_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            result_text = response.content[0].text
        else:
            raise ValueError(f"Unknown provider: {provider}")

        # Parse response
        try:
            result = json.loads(result_text)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response: {e}")
            return {}

        # Convert to issues
        for analysis in result.get("analyses", []):
            scenario_id = analysis.get("scenario_id")
            issues: List[QualityIssue] = []

            if not analysis.get("listener_emotion_plausible", True):
                issues.append(
                    QualityIssue(
                        flag=QualityFlag.IMPLAUSIBLE_LISTENER_EMOTION,
                        severity="major",
                        message="LLM flagged implausible listener emotion",
                        details={
                            "explanation": analysis.get("emotion_issue", ""),
                            "llm_confidence": analysis.get("confidence", 0),
                        },
                    )
                )

            if not analysis.get("vad_consistent", True):
                issues.append(
                    QualityIssue(
                        flag=QualityFlag.VAD_EMOTION_MISMATCH,
                        severity="minor",
                        message="LLM flagged VAD inconsistency",
                        details={
                            "explanation": analysis.get("vad_issue", ""),
                            "llm_confidence": analysis.get("confidence", 0),
                        },
                    )
                )

            if analysis.get("perspective_confused", False):
                issues.append(
                    QualityIssue(
                        flag=QualityFlag.SPEAKER_LISTENER_CONFUSION,
                        severity="major",
                        message="LLM detected possible speaker/listener confusion",
                        details={
                            "explanation": analysis.get("perspective_issue", ""),
                            "llm_confidence": analysis.get("confidence", 0),
                        },
                    )
                )

            if issues:
                issues_by_scenario[scenario_id] = issues

        return issues_by_scenario

    def _build_batch_prompt(
        self,
        batch: List[tuple[ScenarioData, List[AnnotationData], PragmaticSubtype]],
    ) -> str:
        """Build prompt for LLM batch check."""
        from collections import Counter

        scenarios_text: List[str] = []

        for scenario, annotations, subtype in batch:
            # Get majority/consensus annotation
            emotions = [a.plutchik_emotion for a in annotations if a.plutchik_emotion]
            majority_emotion = Counter(emotions).most_common(1)[0][0] if emotions else "unknown"

            # Average VAD
            if annotations:
                v_vals = [a.get_vad_numeric(self.config)["v"] for a in annotations]
                a_vals = [a.get_vad_numeric(self.config)["a"] for a in annotations]
                d_vals = [a.get_vad_numeric(self.config)["d"] for a in annotations]
                avg_v = sum(v_vals) / len(v_vals)
                avg_a = sum(a_vals) / len(a_vals)
                avg_d = sum(d_vals) / len(d_vals)
            else:
                avg_v = avg_a = avg_d = 0.0

            scenario_text = f"""
SCENARIO {scenario.scenario_id} (Subtype: {subtype.value}):
Situation: {scenario.situation}
Speaker ({scenario.speaker_role}) says to Listener ({scenario.listener_role}):
"{scenario.utterance}"

Annotated LISTENER Response:
- Emotion: {majority_emotion}
- Valence (avg): {avg_v:.1f}
- Arousal (avg): {avg_a:.1f}
- Dominance (avg): {avg_d:.1f}
"""
            scenarios_text.append(scenario_text)

        return f"""Analyze the following {len(batch)} annotated scenarios.

Remember: The emotion/VAD labels describe the LISTENER's response to hearing the utterance, NOT the speaker's emotion.

{"".join(scenarios_text)}

Respond with JSON:
{{
    "analyses": [
        {{
            "scenario_id": <id>,
            "listener_emotion_plausible": true/false,
            "emotion_issue": "<explanation if false>",
            "vad_consistent": true/false,
            "vad_issue": "<explanation if false>",
            "perspective_confused": true/false,
            "perspective_issue": "<explanation if true>",
            "confidence": <0.0-1.0>
        }}
    ]
}}"""

    def filter_llm_issues_with_corroboration(
        self,
        llm_issues: Dict[int, List[QualityIssue]],
        existing_issues: Dict[tuple, List[QualityIssue]],
        require_corroboration: bool = True,
    ) -> Dict[int, List[QualityIssue]]:
        """
        Filter LLM-flagged issues to only include those with corroborating evidence.

        This prevents over-reliance on potentially unreliable LLM judgments.
        An LLM flag is kept if:
        1. require_corroboration is False, OR
        2. The scenario has at least one corroborating deterministic issue

        Args:
            llm_issues: Dict mapping scenario_id -> LLM-generated issues
            existing_issues: Dict mapping (subtype, scenario_id) -> existing issues
            require_corroboration: If True, only keep LLM issues with evidence

        Returns:
            Filtered dict of LLM issues (only those with corroboration)
        """
        if not require_corroboration:
            return llm_issues

        filtered: Dict[int, List[QualityIssue]] = {}
        corroborated_count = 0
        filtered_out_count = 0

        for scenario_id, issues in llm_issues.items():
            # Find existing issues for this scenario (check all subtypes)
            has_corroboration = False
            corroborating_evidence: List[str] = []

            for key, key_issues in existing_issues.items():
                if key[1] == scenario_id:
                    for issue in key_issues:
                        if issue.flag in CORROBORATING_FLAGS:
                            has_corroboration = True
                            corroborating_evidence.append(f"{issue.flag.value}: {issue.message}")

            if has_corroboration:
                # Add corroboration info to the LLM issues
                for issue in issues:
                    issue.details["corroborated"] = True
                    issue.details["corroborating_evidence"] = corroborating_evidence
                filtered[scenario_id] = issues
                corroborated_count += len(issues)
            else:
                # Downgrade uncorroborated LLM issues to info severity
                downgraded_issues: List[QualityIssue] = []
                for issue in issues:
                    downgraded = QualityIssue(
                        flag=issue.flag,
                        severity="info",  # Downgrade to info
                        message=f"[Uncorroborated] {issue.message}",
                        details={
                            **issue.details,
                            "corroborated": False,
                            "original_severity": issue.severity,
                            "downgrade_reason": "No deterministic evidence supports this LLM judgment",
                        },
                    )
                    downgraded_issues.append(downgraded)
                filtered[scenario_id] = downgraded_issues
                filtered_out_count += len(issues)

        logger.info(
            f"LLM corroboration: {corroborated_count} issues corroborated, "
            f"{filtered_out_count} downgraded to info"
        )

        return filtered

    def get_corroboration_summary(
        self,
        llm_issues: Dict[int, List[QualityIssue]],
        existing_issues: Dict[tuple, List[QualityIssue]],
    ) -> Dict[str, Any]:
        """
        Generate summary statistics about LLM corroboration.

        Args:
            llm_issues: Raw LLM issues before filtering
            existing_issues: All deterministic issues

        Returns:
            Summary dict with corroboration statistics
        """
        total_llm = sum(len(issues) for issues in llm_issues.values())
        corroborated = 0
        uncorroborated = 0
        by_flag: Dict[str, Dict[str, int]] = {}

        for scenario_id, issues in llm_issues.items():
            # Check for corroboration
            has_corroboration = False
            for key, key_issues in existing_issues.items():
                if key[1] == scenario_id:
                    for issue in key_issues:
                        if issue.flag in CORROBORATING_FLAGS:
                            has_corroboration = True
                            break

            for issue in issues:
                flag_name = issue.flag.value
                if flag_name not in by_flag:
                    by_flag[flag_name] = {"corroborated": 0, "uncorroborated": 0}

                if has_corroboration:
                    corroborated += 1
                    by_flag[flag_name]["corroborated"] += 1
                else:
                    uncorroborated += 1
                    by_flag[flag_name]["uncorroborated"] += 1

        return {
            "total_llm_issues": total_llm,
            "corroborated_count": corroborated,
            "uncorroborated_count": uncorroborated,
            "corroboration_rate": round(corroborated / max(1, total_llm), 3),
            "by_flag": by_flag,
        }
