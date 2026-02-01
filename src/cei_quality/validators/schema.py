"""
Stage 1A: Schema Validation

Validates the structure and completeness of Label Studio JSON exports.

Checks performed:
- Required top-level fields present (id, data, annotations)
- Required data fields present (situation, utterance, roles)
- Required annotation labels present (emotion, VAD, confidence)
- Data types are correct
- No empty/null values in required fields
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, TYPE_CHECKING

from cei_quality.models import QualityFlag, QualityIssue

if TYPE_CHECKING:
    from cei_quality.config import CEIConfig

logger = logging.getLogger(__name__)


class SchemaValidator:
    """
    Stage 1A: Validate Label Studio JSON export schema.

    This validator ensures that exported records have the expected
    structure and all required fields are present and non-empty.

    Example:
        >>> validator = SchemaValidator(config)
        >>> issues = validator.validate_record(record, "file.json")
        >>> for issue in issues:
        ...     print(f"{issue.severity}: {issue.message}")
    """

    def __init__(self, config: CEIConfig) -> None:
        """
        Initialize schema validator.

        Args:
            config: CEI configuration instance
        """
        self.config = config
        self._ls_fields = config.labelstudio

    def validate_record(self, record: Dict[str, Any], file_path: str) -> List[QualityIssue]:
        """
        Validate a single Label Studio record.

        Args:
            record: Parsed JSON record
            file_path: Source file path for error messages

        Returns:
            List of quality issues found
        """
        issues: List[QualityIssue] = []

        # Check top-level structure
        issues.extend(self._validate_top_level(record, file_path))

        # If we don't have data, can't continue
        if "data" not in record:
            return issues

        # Check data fields
        task_id = record.get("id", "unknown")
        issues.extend(self._validate_data_fields(record["data"], task_id))

        # Check annotations
        if "annotations" in record:
            issues.extend(self._validate_annotations(record["annotations"], task_id))

        return issues

    def _validate_top_level(self, record: Dict[str, Any], file_path: str) -> List[QualityIssue]:
        """Validate top-level required fields."""
        issues: List[QualityIssue] = []

        required_fields = {"id", "data", "annotations"}

        for field in required_fields:
            if field not in record:
                issues.append(
                    QualityIssue(
                        flag=QualityFlag.MISSING_REQUIRED_FIELD,
                        severity="critical",
                        message=f"Missing top-level field: '{field}'",
                        details={"field": field, "file": file_path},
                    )
                )

        # Check id is valid
        if "id" in record:
            task_id = record["id"]
            if not isinstance(task_id, (int, str)):
                issues.append(
                    QualityIssue(
                        flag=QualityFlag.INVALID_DATA_TYPE,
                        severity="major",
                        message=f"Invalid type for 'id': {type(task_id).__name__}",
                        details={
                            "field": "id",
                            "expected": "int or str",
                            "actual": type(task_id).__name__,
                        },
                    )
                )

        return issues

    def _validate_data_fields(self, data: Dict[str, Any], task_id: Any) -> List[QualityIssue]:
        """Validate scenario data fields."""
        issues: List[QualityIssue] = []

        # Map of field names to their config keys
        field_mapping = {
            self._ls_fields.data_fields.situation: "situation",
            self._ls_fields.data_fields.utterance: "utterance",
            self._ls_fields.data_fields.speaker_role: "speaker_role",
            self._ls_fields.data_fields.listener_role: "listener_role",
            self._ls_fields.data_fields.scenario_id: "scenario_id",
        }

        for field_name, display_name in field_mapping.items():
            # Check field exists
            if field_name not in data:
                severity = "critical" if display_name == "scenario_id" else "major"
                issues.append(
                    QualityIssue(
                        flag=QualityFlag.MISSING_REQUIRED_FIELD,
                        severity=severity,
                        message=f"Missing data field: '{field_name}' ({display_name})",
                        details={"field": field_name, "task_id": task_id},
                    )
                )
                continue

            value = data[field_name]

            # Check for null/None
            if value is None:
                issues.append(
                    QualityIssue(
                        flag=QualityFlag.EMPTY_VALUE,
                        severity="major",
                        message=f"Null value for data field: '{field_name}'",
                        details={"field": field_name, "task_id": task_id},
                    )
                )
                continue

            # Check for empty strings (except scenario_id which can be 0)
            if display_name != "scenario_id":
                if isinstance(value, str) and not value.strip():
                    issues.append(
                        QualityIssue(
                            flag=QualityFlag.EMPTY_VALUE,
                            severity="major",
                            message=f"Empty string for data field: '{field_name}'",
                            details={"field": field_name, "task_id": task_id},
                        )
                    )

        # Validate scenario_id is a valid integer
        scenario_id_field = self._ls_fields.data_fields.scenario_id
        if scenario_id_field in data:
            scenario_id = data[scenario_id_field]
            if not isinstance(scenario_id, int):
                try:
                    int(scenario_id)
                except (ValueError, TypeError):
                    issues.append(
                        QualityIssue(
                            flag=QualityFlag.INVALID_DATA_TYPE,
                            severity="critical",
                            message=f"Invalid scenario_id type: {type(scenario_id).__name__}",
                            details={
                                "field": scenario_id_field,
                                "value": scenario_id,
                                "task_id": task_id,
                            },
                        )
                    )

        return issues

    def _validate_annotations(
        self, annotations: List[Dict[str, Any]], task_id: Any
    ) -> List[QualityIssue]:
        """Validate annotations array."""
        issues: List[QualityIssue] = []

        # Check for empty annotations
        if not annotations:
            issues.append(
                QualityIssue(
                    flag=QualityFlag.NO_ANNOTATIONS,
                    severity="critical",
                    message="No annotations present in record",
                    details={"task_id": task_id},
                )
            )
            return issues

        # Validate each annotation
        for idx, annotation in enumerate(annotations):
            issues.extend(self._validate_single_annotation(annotation, task_id, idx))

        return issues

    def _validate_single_annotation(
        self, annotation: Dict[str, Any], task_id: Any, idx: int
    ) -> List[QualityIssue]:
        """Validate a single annotation object."""
        issues: List[QualityIssue] = []

        # Check for result field
        if "result" not in annotation:
            issues.append(
                QualityIssue(
                    flag=QualityFlag.MISSING_REQUIRED_FIELD,
                    severity="critical",
                    message=f"Annotation {idx} missing 'result' field",
                    details={"task_id": task_id, "annotation_idx": idx},
                )
            )
            return issues

        result = annotation["result"]

        if not isinstance(result, list):
            issues.append(
                QualityIssue(
                    flag=QualityFlag.INVALID_DATA_TYPE,
                    severity="critical",
                    message=f"Annotation 'result' is not a list: {type(result).__name__}",
                    details={"task_id": task_id, "annotation_idx": idx},
                )
            )
            return issues

        # Extract labeled fields from result
        found_labels: set[str] = set()

        for result_item in result:
            from_name = result_item.get("from_name", "")
            found_labels.add(from_name)

            # Check value structure
            value = result_item.get("value", {})
            if not value:
                issues.append(
                    QualityIssue(
                        flag=QualityFlag.EMPTY_VALUE,
                        severity="major",
                        message=f"Empty value for label '{from_name}'",
                        details={"task_id": task_id, "label": from_name, "annotation_idx": idx},
                    )
                )
                continue

            # Check choices are present and non-empty
            choices = value.get("choices", [])
            rating = value.get("rating")

            if not choices and rating is None:
                issues.append(
                    QualityIssue(
                        flag=QualityFlag.EMPTY_VALUE,
                        severity="major",
                        message=f"No choices or rating for label '{from_name}'",
                        details={"task_id": task_id, "label": from_name, "annotation_idx": idx},
                    )
                )

        # Check all required label fields are present
        required_labels = {
            self._ls_fields.label_fields.plutchik_emotion,
            self._ls_fields.label_fields.valence,
            self._ls_fields.label_fields.arousal,
            self._ls_fields.label_fields.dominance,
            self._ls_fields.label_fields.confidence,
        }

        missing_labels = required_labels - found_labels

        for label in missing_labels:
            issues.append(
                QualityIssue(
                    flag=QualityFlag.INCOMPLETE_ANNOTATION,
                    severity="major",
                    message=f"Missing annotation label: '{label}'",
                    details={"task_id": task_id, "label": label, "annotation_idx": idx},
                )
            )

        return issues

    def validate_file(
        self, records: List[Dict[str, Any]], file_path: str
    ) -> Dict[int, List[QualityIssue]]:
        """
        Validate all records in a file.

        Args:
            records: List of parsed JSON records
            file_path: Source file path

        Returns:
            Dict mapping scenario_id to list of issues
        """
        issues_by_scenario: Dict[int, List[QualityIssue]] = {}

        for record in records:
            issues = self.validate_record(record, file_path)

            # Get scenario ID
            data = record.get("data", {})
            scenario_id_field = self._ls_fields.data_fields.scenario_id
            scenario_id = data.get(scenario_id_field, 0)

            if isinstance(scenario_id, str):
                try:
                    scenario_id = int(scenario_id)
                except ValueError:
                    scenario_id = 0

            if scenario_id not in issues_by_scenario:
                issues_by_scenario[scenario_id] = []

            issues_by_scenario[scenario_id].extend(issues)

        return issues_by_scenario
