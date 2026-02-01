"""
CEI Quality Pipeline CLI.

Command-line interface for running the quality filtering pipeline.

Usage:
    cei-quality run --config config/config.yml
    cei-quality run --data-dir ./scenarios/openai/gpt-5-mini/gold-gpt5mini
    cei-quality validate --file path/to/file.json
    cei-quality report --input outputs/quality/sampling_plan.json
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn

from cei_quality import __version__
from cei_quality.config import CEIConfig, load_config
from cei_quality.pipeline import CEIQualityPipeline

console = Console()


def setup_logging(level: str, rich_console: bool = True) -> None:
    """Configure logging with optional rich formatting."""
    log_level = getattr(logging, level.upper(), logging.INFO)

    if rich_console:
        logging.basicConfig(
            level=log_level,
            format="%(message)s",
            datefmt="[%X]",
            handlers=[RichHandler(console=console, rich_tracebacks=True)],
        )
    else:
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )


@click.group()
@click.version_option(version=__version__, prog_name="cei-quality")
def main() -> None:
    """
    CEI Quality Pipeline - Quality filtering for annotation data.

    This tool validates Label Studio annotation exports and generates
    stratified sampling plans for human expert review.

    \b
    Examples:
        cei-quality run --config config/config.yml
        cei-quality run --data-dir ./data --output-dir ./outputs
        cei-quality validate --file annotations.json
    """
    pass


@main.command()
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to configuration YAML file.",
)
@click.option(
    "--data-dir",
    "-d",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Override data directory from config.",
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(path_type=Path),
    default=None,
    help="Override output directory from config.",
)
@click.option(
    "--llm/--no-llm",
    default=None,
    help="Enable/disable LLM plausibility checks.",
)
@click.option(
    "--log-level",
    "-l",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False),
    default="INFO",
    help="Logging verbosity level.",
)
@click.option(
    "--quiet",
    "-q",
    is_flag=True,
    help="Suppress console output (still writes to files).",
)
def run(
    config: Optional[Path],
    data_dir: Optional[Path],
    output_dir: Optional[Path],
    llm: Optional[bool],
    log_level: str,
    quiet: bool,
) -> None:
    """
    Run the complete quality filtering pipeline.

    Validates annotation data and generates sampling plans for human review.

    \b
    Examples:
        # Run with config file
        cei-quality run --config config/config.yml

        # Run with overrides
        cei-quality run -d ./data -o ./outputs --no-llm

        # Run in quiet mode
        cei-quality run -c config.yml -q
    """
    setup_logging(log_level, rich_console=not quiet)
    logger = logging.getLogger("cei_quality.cli")

    # Load configuration
    try:
        overrides = {}
        if data_dir:
            overrides["paths.data_dir"] = str(data_dir)
        if output_dir:
            overrides["paths.output_dir"] = str(output_dir)
        if llm is not None:
            overrides["llm.enabled"] = llm

        cfg = load_config(
            config_path=config,
            project_root=Path.cwd(),
            overrides=overrides if overrides else None,
        )

        if not quiet:
            console.print(f"[bold blue]CEI Quality Pipeline v{__version__}[/bold blue]")
            console.print(f"Config: {config or 'defaults'}")
            console.print(f"Data dir: {cfg.get_resolved_paths().data_dir}")
            console.print(f"Output dir: {cfg.get_resolved_paths().output_dir}")
            console.print()

    except Exception as e:
        console.print(f"[red]Error loading configuration:[/red] {e}")
        sys.exit(1)

    # Run pipeline
    try:
        pipeline = CEIQualityPipeline(cfg)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console if not quiet else Console(quiet=True),
        ) as progress:
            task = progress.add_task("Running quality pipeline...", total=None)
            sampling_plan = pipeline.run()
            progress.update(task, completed=True)

        # Summary
        if not quiet:
            console.print()
            console.print("[bold green]Pipeline completed successfully![/bold green]")
            console.print(f"  Total scenarios: {sampling_plan.total_scenarios}")
            console.print(f"  Scenarios flagged: {sampling_plan.scenarios_flagged}")
            console.print(f"  Selected for review: {sampling_plan.sample_size}")
            console.print(f"  Output: {cfg.get_resolved_paths().output_dir}")

    except Exception as e:
        logger.exception("Pipeline failed")
        console.print(f"[red]Pipeline error:[/red] {e}")
        sys.exit(1)


@main.command()
@click.argument("file", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to configuration YAML file.",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Show detailed issue information.",
)
def validate(file: Path, config: Optional[Path], verbose: bool) -> None:
    """
    Validate a single annotation file.

    Runs schema and consistency checks on a single file without
    running the full pipeline.

    \b
    Example:
        cei-quality validate annotations.json
        cei-quality validate --verbose file.json
    """
    setup_logging("INFO")

    import json
    from cei_quality.validators import SchemaValidator, WithinFileValidator
    from cei_quality.models import AnnotatorFile

    # Load config
    cfg = load_config(config_path=config, project_root=Path.cwd())

    console.print(f"[bold]Validating:[/bold] {file}")
    console.print()

    # Load file
    try:
        with open(file) as f:
            data = json.load(f)

        if isinstance(data, list):
            records = data
        elif isinstance(data, dict) and "tasks" in data:
            records = data["tasks"]
        else:
            records = [data]

        console.print(f"Loaded {len(records)} records")

    except Exception as e:
        console.print(f"[red]Error loading file:[/red] {e}")
        sys.exit(1)

    # Run validators
    af = AnnotatorFile.from_filename(file)
    af.records = records

    # Schema validation
    schema_validator = SchemaValidator(cfg)
    schema_issues = schema_validator.validate_file(records, str(file))

    # Consistency validation
    consistency_validator = WithinFileValidator(cfg)
    consistency_issues, file_report = consistency_validator.validate_file(af)

    # Report results
    total_issues = sum(len(v) for v in schema_issues.values())
    total_issues += sum(len(v) for v in consistency_issues.values())

    if total_issues == 0:
        console.print("[green]✓ No issues found![/green]")
    else:
        console.print(f"[yellow]Found {total_issues} issues:[/yellow]")

        # Count by severity
        severities = {"critical": 0, "major": 0, "minor": 0, "info": 0}
        all_issues = []

        for issues in list(schema_issues.values()) + list(consistency_issues.values()):
            for issue in issues:
                severities[issue.severity] += 1
                all_issues.append(issue)

        for severity, count in severities.items():
            if count > 0:
                color = {
                    "critical": "red",
                    "major": "yellow",
                    "minor": "blue",
                    "info": "dim",
                }[severity]
                console.print(f"  [{color}]{severity}:[/{color}] {count}")

        if verbose:
            console.print()
            console.print("[bold]Issue Details:[/bold]")
            for issue in all_issues[:20]:  # Show first 20
                color = {
                    "critical": "red",
                    "major": "yellow",
                    "minor": "blue",
                    "info": "dim",
                }[issue.severity]
                console.print(f"  [{color}]{issue.flag.value}:[/{color}] {issue.message}")

    # File report
    if file_report.needs_full_review:
        console.print()
        console.print(f"[red]⚠ File needs full review:[/red] {file_report.review_reason}")


@main.command()
@click.argument("input_file", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--format",
    "-f",
    type=click.Choice(["text", "json", "markdown"]),
    default="text",
    help="Output format.",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    default=None,
    help="Output file (default: stdout).",
)
def report(input_file: Path, format: str, output: Optional[Path]) -> None:
    """
    Generate a report from pipeline results.

    Reads a sampling_plan.json or quality reports file and generates
    a formatted report.

    \b
    Example:
        cei-quality report outputs/quality/sampling_plan.json
        cei-quality report -f markdown -o report.md sampling_plan.json
    """
    import json

    with open(input_file) as f:
        data = json.load(f)

    if format == "json":
        output_text = json.dumps(data, indent=2)
    elif format == "markdown":
        output_text = _generate_markdown_report(data)
    else:
        output_text = _generate_text_report(data)

    if output:
        with open(output, "w") as f:
            f.write(output_text)
        console.print(f"Report written to: {output}")
    else:
        console.print(output_text)


def _generate_text_report(data: dict) -> str:
    """Generate text report from sampling plan data."""
    lines = [
        "=" * 70,
        "CEI DATA QUALITY REPORT",
        "=" * 70,
        "",
    ]

    if "summary" in data:
        summary = data["summary"]
        lines.extend(
            [
                "SUMMARY",
                "-" * 40,
                f"Total Scenarios:     {summary.get('total_scenarios', 'N/A')}",
                f"Scenarios Flagged:   {summary.get('scenarios_flagged', 'N/A')}",
                f"Sample Size:         {summary.get('sample_size', 'N/A')}",
                f"Mandatory Reviews:   {summary.get('mandatory_count', 'N/A')}",
                f"Stratified Sample:   {summary.get('stratified_count', 'N/A')}",
                "",
            ]
        )

    if "agreement" in data:
        agreement = data["agreement"]
        lines.extend(
            [
                "INTER-ANNOTATOR AGREEMENT",
                "-" * 40,
                f"Overall Fleiss' κ:   {agreement.get('overall_fleiss_kappa', 'N/A'):.3f}",
            ]
        )
        for subtype, kappa in agreement.get("by_subtype", {}).items():
            lines.append(f"  {subtype}: {kappa:.3f}")
        lines.append("")

    lines.extend(
        [
            "=" * 70,
        ]
    )

    return "\n".join(lines)


def _generate_markdown_report(data: dict) -> str:
    """Generate Markdown report from sampling plan data."""
    lines = [
        "# CEI Data Quality Report",
        "",
        f"Generated: {datetime.now().isoformat()}",
        "",
        "## Summary",
        "",
    ]

    if "summary" in data:
        summary = data["summary"]
        lines.extend(
            [
                "| Metric | Value |",
                "|--------|-------|",
                f"| Total Scenarios | {summary.get('total_scenarios', 'N/A')} |",
                f"| Scenarios Flagged | {summary.get('scenarios_flagged', 'N/A')} |",
                f"| Sample Size | {summary.get('sample_size', 'N/A')} |",
                f"| Mandatory Reviews | {summary.get('mandatory_count', 'N/A')} |",
                f"| Stratified Sample | {summary.get('stratified_count', 'N/A')} |",
                "",
            ]
        )

    if "agreement" in data:
        lines.extend(
            [
                "## Inter-Annotator Agreement",
                "",
                "| Subtype | Fleiss' κ |",
                "|---------|-----------|",
            ]
        )
        for subtype, kappa in data["agreement"].get("by_subtype", {}).items():
            lines.append(f"| {subtype} | {kappa:.3f} |")
        lines.append("")

    return "\n".join(lines)


@main.command()
def init() -> None:
    """
    Initialize a new project with default configuration.

    Creates config/config.yml and directory structure.
    If config.yml already exists, creates a timestamped backup first.
    """
    from datetime import datetime

    config_dir = Path("config")
    config_dir.mkdir(exist_ok=True)

    config_file = config_dir / "config.yml"

    # Backup existing config if present
    if config_file.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = config_dir / f"config_{timestamp}.yml"

        # Copy existing config to backup
        import shutil
        shutil.copy(config_file, backup_file)
        console.print(f"[yellow]Backed up:[/yellow] {config_file} → {backup_file}")

    # Write default config with new paths
    default_config = """# CEI Quality Pipeline Configuration
# See documentation for all options

project:
  name: "My CEI Project"
  version: "1.0.0"

paths:
  data_dir: "input/openai/gpt-5-mini/gold-gpt5mini"
  output_dir: "output/openai/gpt-5-mini/gold-gpt5mini"
  reports_dir: "reports"
  logs_dir: "logs"

schema:
  scenarios_per_subtype: 60
  annotators_per_subtype: 3

# Quality thresholds for differentiation
thresholds:
  min_lead_time_seconds: 5.0
  impossibly_fast_seconds: 3.0
  unusually_slow_seconds: 600.0
  dwell_time_outlier_z: 2.0
  fleiss_kappa_warning: 0.3
  fleiss_kappa_acceptable: 0.4
  fleiss_kappa_good: 0.6
  mandatory_review_score: 0.80
  flag_review_score: 0.90

# Human review settings
review:
  show_all_annotators: true
  max_review_items: 100
  priority_critical: 90
  priority_high: 70
  priority_medium: 50

logging:
  level: "INFO"
  rich_console: true

llm:
  enabled: false
  provider: "openai"
  model: "gpt-5-mini"

# Valid label values for each dimension (with numeric mappings for VAD)
valid_labels:
  plutchik_emotions:
    - joy
    - trust
    - fear
    - surprise
    - sadness
    - disgust
    - anger
    - anticipation
  valence:
    - value: "very pleasant"
      numeric: 1.0
    - value: "pleasant"
      numeric: 0.67
    - value: "slightly pleasant"
      numeric: 0.33
    - value: "neutral"
      numeric: 0.0
    - value: "slightly unpleasant"
      numeric: -0.33
    - value: "unpleasant"
      numeric: -0.67
    - value: "very unpleasant"
      numeric: -1.0
  arousal:
    - value: "very excited"
      numeric: 1.0
    - value: "excited"
      numeric: 0.67
    - value: "slightly excited"
      numeric: 0.33
    - value: "neutral"
      numeric: 0.0
    - value: "slightly calm"
      numeric: -0.33
    - value: "calm"
      numeric: -0.67
    - value: "very calm"
      numeric: -1.0
  dominance:
    - value: "very in control"
      numeric: 1.0
    - value: "in control"
      numeric: 0.67
    - value: "slightly in control"
      numeric: 0.33
    - value: "neutral"
      numeric: 0.0
    - value: "slightly controlled"
      numeric: -0.33
    - value: "controlled"
      numeric: -0.67
    - value: "very controlled"
      numeric: -1.0
  confidence:
    - value: "very confident"
      numeric: 1.0
    - value: "confident"
      numeric: 0.67
    - value: "mildly confident"
      numeric: 0.33
    - value: "neutral"
      numeric: 0.0
    - value: "mildly uncertain"
      numeric: -0.33
    - value: "uncertain"
      numeric: -0.67
    - value: "very uncertain"
      numeric: -1.0

# Expected VAD ranges for each Plutchik emotion
emotion_vad_profiles:
  joy:
    valence: [0.3, 1.0]
    arousal: [-0.3, 0.7]
    dominance: [0.0, 1.0]
  trust:
    valence: [0.0, 0.8]
    arousal: [-0.6, 0.3]
    dominance: [-0.3, 0.6]
  fear:
    valence: [-1.0, -0.2]
    arousal: [0.2, 1.0]
    dominance: [-1.0, -0.1]
  surprise:
    valence: [-0.5, 0.7]
    arousal: [0.3, 1.0]
    dominance: [-0.6, 0.4]
  sadness:
    valence: [-1.0, -0.2]
    arousal: [-1.0, 0.0]
    dominance: [-1.0, -0.1]
  disgust:
    valence: [-1.0, -0.3]
    arousal: [-0.3, 0.6]
    dominance: [0.0, 0.8]
  anger:
    valence: [-1.0, -0.3]
    arousal: [0.3, 1.0]
    dominance: [0.2, 1.0]
  anticipation:
    valence: [0.0, 0.7]
    arousal: [0.0, 0.7]
    dominance: [-0.2, 0.6]

quality:
  vad_tolerance: 0.3
"""

    with open(config_file, "w") as f:
        f.write(default_config)

    console.print(f"[green]Created:[/green] {config_file}")
    console.print()
    console.print("Next steps:")
    console.print("  1. Review/edit config/config.yml for your data paths")
    console.print("  2. Place annotation files in input/ directory")
    console.print("  3. Run: cei-quality run --config config/config.yml")


if __name__ == "__main__":
    main()
