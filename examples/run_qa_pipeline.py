#!/usr/bin/env python3
"""Example: Running the CEI QA pipeline."""

import subprocess
import sys
from pathlib import Path


def run_qa_pipeline():
    """Run the 4-level QA pipeline on gold annotations."""
    config_path = Path(__file__).parent.parent / "config" / "config.yml"

    if not config_path.exists():
        print(f"Error: Config file not found at {config_path}")
        sys.exit(1)

    print("Running CEI Quality Pipeline...")
    print(f"Config: {config_path}")
    print()

    # Run the CLI
    result = subprocess.run(
        ["cei-quality", "run", "--config", str(config_path)],
        capture_output=False
    )

    return result.returncode


def main():
    """Main entry point."""
    print("=" * 60)
    print("CEI Benchmark Quality Assurance Pipeline")
    print("=" * 60)
    print()
    print("This pipeline performs 4-level validation:")
    print("  Level 1A: Schema validation (JSON structure)")
    print("  Level 1B: Within-file consistency (timing, straight-lining)")
    print("  Level 1C: Inter-annotator agreement (Fleiss' kappa)")
    print("  Level 1D: Semantic plausibility (VAD-emotion consistency)")
    print()

    return_code = run_qa_pipeline()

    if return_code == 0:
        print("\nPipeline completed successfully!")
        print("Reports saved to: data/qa_reports/")
    else:
        print(f"\nPipeline failed with return code: {return_code}")

    sys.exit(return_code)


if __name__ == "__main__":
    main()
