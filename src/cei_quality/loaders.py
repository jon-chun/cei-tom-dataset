"""
Data Loading for CEI Quality Pipeline.

Handles loading and parsing of Label Studio JSON exports.

Supports:
- JSON files (array of records)
- JSONL files (one record per line)
- CSV files (as backup format)
"""

from __future__ import annotations

import csv
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from cei_quality.models import (
    AnnotationData,
    AnnotatorFile,
    PragmaticSubtype,
    QualityIssue,
    QualityFlag,
    ScenarioData,
)

if TYPE_CHECKING:
    from cei_quality.config import CEIConfig

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Load and parse Label Studio JSON exports.

    This class handles:
    - File discovery based on naming patterns
    - JSON/JSONL parsing
    - Record extraction and validation
    - Organizing data by subtype and annotator

    Example:
        >>> loader = DataLoader(config)
        >>> organized = loader.load_all()
        >>> for subtype, annotators in organized.items():
        ...     print(f"{subtype.value}: {len(annotators)} annotators")
    """

    def __init__(self, config: CEIConfig) -> None:
        """
        Initialize data loader.

        Args:
            config: CEI configuration instance
        """
        self.config = config
        self.data_dir = config.get_resolved_paths().data_dir
        self._ls_fields = config.labelstudio

    def discover_files(self) -> List[AnnotatorFile]:
        """
        Discover all annotation files in the data directory.

        Returns:
            List of AnnotatorFile metadata objects
        """
        files: List[AnnotatorFile] = []

        if not self.data_dir.exists():
            logger.error(f"Data directory does not exist: {self.data_dir}")
            return files

        # Look for supported file types
        extensions = self.config.schema_config.file_extensions

        for ext in extensions:
            pattern = f"*{ext}"
            for file_path in sorted(self.data_dir.glob(pattern)):
                try:
                    af = AnnotatorFile.from_filename(file_path)
                    files.append(af)
                    logger.info(
                        f"Discovered: {file_path.name} -> {af.annotator_name} / {af.subtype.value}"
                    )
                except Exception as e:
                    logger.warning(f"Failed to parse filename {file_path}: {e}")

        logger.info(f"Discovered {len(files)} annotation files")
        return files

    def load_file(self, af: AnnotatorFile) -> Tuple[List[Dict[str, Any]], List[QualityIssue]]:
        """
        Load records from a single file.

        Args:
            af: AnnotatorFile to load

        Returns:
            Tuple of (records list, loading issues)
        """
        issues: List[QualityIssue] = []
        records: List[Dict[str, Any]] = []

        file_path = af.file_path

        if not file_path.exists():
            issues.append(
                QualityIssue(
                    flag=QualityFlag.MALFORMED_JSON,
                    severity="critical",
                    message=f"File does not exist: {file_path}",
                    details={"file": str(file_path)},
                )
            )
            return records, issues

        try:
            suffix = file_path.suffix.lower()

            if suffix == ".json":
                records, issues = self._load_json(file_path)
            elif suffix == ".jsonl":
                records, issues = self._load_jsonl(file_path)
            elif suffix == ".csv":
                records, issues = self._load_csv(file_path)
            else:
                issues.append(
                    QualityIssue(
                        flag=QualityFlag.MALFORMED_JSON,
                        severity="critical",
                        message=f"Unsupported file type: {suffix}",
                        details={"file": str(file_path)},
                    )
                )

        except Exception as e:
            issues.append(
                QualityIssue(
                    flag=QualityFlag.MALFORMED_JSON,
                    severity="critical",
                    message=f"Failed to read file: {e}",
                    details={"file": str(file_path), "error": str(e)},
                )
            )

        af.records = records
        logger.info(f"Loaded {len(records)} records from {file_path.name}")

        return records, issues

    def _load_json(self, file_path: Path) -> Tuple[List[Dict[str, Any]], List[QualityIssue]]:
        """Load JSON file (array or single object)."""
        issues: List[QualityIssue] = []

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        try:
            data = json.loads(content)
        except json.JSONDecodeError as e:
            issues.append(
                QualityIssue(
                    flag=QualityFlag.MALFORMED_JSON,
                    severity="critical",
                    message=f"JSON parse error: {e}",
                    details={"file": str(file_path), "error": str(e)},
                )
            )
            return [], issues

        if isinstance(data, list):
            return data, issues
        elif isinstance(data, dict):
            # Could be wrapped format
            if "tasks" in data:
                return data["tasks"], issues
            else:
                return [data], issues
        else:
            issues.append(
                QualityIssue(
                    flag=QualityFlag.MALFORMED_JSON,
                    severity="critical",
                    message=f"Unexpected JSON root type: {type(data).__name__}",
                    details={"file": str(file_path)},
                )
            )
            return [], issues

    def _load_jsonl(self, file_path: Path) -> Tuple[List[Dict[str, Any]], List[QualityIssue]]:
        """Load JSONL file (one record per line)."""
        issues: List[QualityIssue] = []
        records: List[Dict[str, Any]] = []

        with open(file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    record = json.loads(line)
                    records.append(record)
                except json.JSONDecodeError as e:
                    issues.append(
                        QualityIssue(
                            flag=QualityFlag.MALFORMED_JSON,
                            severity="major",
                            message=f"JSON parse error at line {line_num}",
                            details={"file": str(file_path), "line": line_num, "error": str(e)},
                        )
                    )

        return records, issues

    def _load_csv(self, file_path: Path) -> Tuple[List[Dict[str, Any]], List[QualityIssue]]:
        """Load CSV file (backup format)."""
        issues: List[QualityIssue] = []
        records: List[Dict[str, Any]] = []

        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                records.append(dict(row))

        # Note: CSV format may need special handling to match JSON structure
        if records:
            issues.append(
                QualityIssue(
                    flag=QualityFlag.MALFORMED_JSON,
                    severity="info",
                    message="CSV format loaded - may need structure conversion",
                    details={"file": str(file_path), "record_count": len(records)},
                )
            )

        return records, issues

    def load_all(self) -> Dict[PragmaticSubtype, Dict[str, AnnotatorFile]]:
        """
        Load all files and organize by subtype.

        Returns:
            Dict mapping subtype -> annotator_name -> AnnotatorFile
        """
        files = self.discover_files()

        organized: Dict[PragmaticSubtype, Dict[str, AnnotatorFile]] = defaultdict(dict)

        for af in files:
            records, issues = self.load_file(af)

            if issues:
                for issue in issues:
                    logger.warning(f"File loading issue: {issue.message}")

            organized[af.subtype][af.annotator_name] = af

        # Log summary
        for subtype in PragmaticSubtype:
            annotators = organized.get(subtype, {})
            expected = self.config.schema_config.annotators_per_subtype
            if len(annotators) != expected:
                logger.warning(
                    f"Subtype {subtype.value}: {len(annotators)} annotators (expected {expected})"
                )

        return organized

    def parse_record(
        self,
        record: Dict[str, Any],
        af: AnnotatorFile,
    ) -> Tuple[Optional[ScenarioData], Optional[AnnotationData], List[QualityIssue]]:
        """
        Parse a single record into scenario and annotation data.

        Args:
            record: Raw JSON record
            af: AnnotatorFile this record came from

        Returns:
            Tuple of (scenario, annotation, issues)
        """
        issues: List[QualityIssue] = []

        # Extract scenario data
        data = record.get("data", {})
        task_id = record.get("id", 0)

        scenario_id_field = self._ls_fields.data_fields.scenario_id
        scenario_id = data.get(scenario_id_field)

        if scenario_id is None:
            issues.append(
                QualityIssue(
                    flag=QualityFlag.MISSING_REQUIRED_FIELD,
                    severity="critical",
                    message=f"Missing {scenario_id_field} field",
                    details={"task_id": task_id},
                )
            )
            return None, None, issues

        # Convert scenario_id to int if needed
        if isinstance(scenario_id, str):
            try:
                scenario_id = int(scenario_id)
            except ValueError:
                issues.append(
                    QualityIssue(
                        flag=QualityFlag.INVALID_DATA_TYPE,
                        severity="critical",
                        message=f"Invalid scenario_id: {scenario_id}",
                        details={"task_id": task_id, "value": scenario_id},
                    )
                )
                return None, None, issues

        # Build scenario
        df = self._ls_fields.data_fields
        scenario = ScenarioData(
            scenario_id=scenario_id,
            task_id=task_id,
            situation=data.get(df.situation, ""),
            utterance=data.get(df.utterance, ""),
            speaker_role=data.get(df.speaker_role, ""),
            listener_role=data.get(df.listener_role, ""),
            source_file=str(af.file_path),
        )

        # Extract annotation
        annotations_list = record.get("annotations", [])
        if not annotations_list:
            issues.append(
                QualityIssue(
                    flag=QualityFlag.NO_ANNOTATIONS,
                    severity="critical",
                    message="No annotations in record",
                    details={"scenario_id": scenario_id, "task_id": task_id},
                )
            )
            return scenario, None, issues

        # Take first annotation
        ann = annotations_list[0]
        result_items = ann.get("result", [])

        # Extract labels from result array
        labels: Dict[str, str] = {}
        for item in result_items:
            from_name = item.get("from_name", "")
            choices = item.get("value", {}).get("choices", [])
            if choices:
                labels[from_name] = choices[0]

        # Build annotation
        lf = self._ls_fields.label_fields
        annotation = AnnotationData(
            scenario_id=scenario_id,
            annotator_id=str(ann.get("completed_by", "")),
            annotator_name=af.annotator_name,
            file_path=str(af.file_path),
            plutchik_emotion=labels.get(lf.plutchik_emotion, ""),
            valence=labels.get(lf.valence, ""),
            arousal=labels.get(lf.arousal, ""),
            dominance=labels.get(lf.dominance, ""),
            confidence=labels.get(lf.confidence, ""),
            lead_time=ann.get("lead_time"),
            created_at=ann.get("created_at"),
        )

        return scenario, annotation, issues
