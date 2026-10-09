#!/usr/bin/env python3
"""
Anonymize CEI Dataset Repository for Blind Review Submission

This script:
1. Renames gold data files to remove annotator names
2. Sanitizes QA reports to remove local paths
3. Updates any internal references to renamed files
4. Generates instructions for creating anonymous git history

Usage:
    python scripts/anonymize_for_submission.py --dry-run  # Preview changes
    python scripts/anonymize_for_submission.py            # Apply changes
"""

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Dict, List, Tuple


# Mapping of annotator names to anonymous IDs
# Format: filename pattern -> (anonymous_id, subtype)
ANNOTATOR_MAPPING = {
    "project-13_anne-duncan": ("annotator-A", "deflection-misdirection"),
    "project-14_tiffanie": ("annotator-B", "deflection-misdirection"),
    "project-15_kirill": ("annotator-C", "deflection-misdirection"),
    "project-16_anna": ("annotator-A", "mixed-signals"),
    "project-17-peter": ("annotator-B", "mixed-signals"),
    "project-18_abhigya": ("annotator-C", "mixed-signals"),
    "project-20_eli": ("annotator-A", "passive-aggression"),
    "project-21_godwin": ("annotator-B", "passive-aggression"),
    "project-28_adrian": ("annotator-C", "passive-aggression"),
    "project-22_hannah": ("annotator-A", "sarcasm-irony"),
    "project-23_andre": ("annotator-B", "sarcasm-irony"),
    "project-24_gwen": ("annotator-C", "sarcasm-irony"),
    "project-25_morty": ("annotator-A", "strategic-politeness"),
    "project-26_wisdom": ("annotator-B", "strategic-politeness"),
    "project-27_mous": ("annotator-C", "strategic-politeness"),
}


def get_repo_root() -> Path:
    """Get repository root directory."""
    script_path = Path(__file__).resolve()
    return script_path.parent.parent


def build_rename_map(gold_dir: Path) -> Dict[Path, Path]:
    """Build mapping of old filenames to new anonymous filenames."""
    rename_map = {}

    for old_path in gold_dir.glob("*.json"):
        old_name = old_path.stem

        # Find matching pattern
        for pattern, (anon_id, subtype) in ANNOTATOR_MAPPING.items():
            if old_name.startswith(pattern):
                new_name = f"{anon_id}_{subtype}.json"
                new_path = gold_dir / new_name
                rename_map[old_path] = new_path
                break
        else:
            print(f"WARNING: No mapping for {old_name}")

    return rename_map


def rename_gold_files(rename_map: Dict[Path, Path], dry_run: bool) -> List[Tuple[str, str]]:
    """Rename gold data files to remove annotator names."""
    changes = []

    for old_path, new_path in rename_map.items():
        changes.append((old_path.name, new_path.name))

        if not dry_run:
            if new_path.exists():
                print(f"  WARNING: {new_path.name} already exists, skipping")
                continue
            shutil.move(old_path, new_path)
            print(f"  Renamed: {old_path.name} -> {new_path.name}")

    return changes


def sanitize_json_paths(file_path: Path, dry_run: bool) -> int:
    """Remove local paths from JSON files. Returns count of replacements."""
    with open(file_path, 'r') as f:
        content = f.read()

    original = content

    # Pattern to match local paths like /Users/jonc/code/cei-tom-benchmark/...
    # Replace with relative path or generic placeholder
    patterns = [
        (r'/Users/[^/]+/code/[^/]+/input/[^"]+/([^"]+\.json)', r'data/gold/\1'),
        (r'/Users/[^/]+/code/[^/]+/[^"]*', 'data/'),
        (r'"/Users/[^"]+/"', '"data/"'),
    ]

    for pattern, replacement in patterns:
        content = re.sub(pattern, replacement, content)

    # Count changes
    n_changes = len(re.findall(r'/Users/', original))

    if not dry_run and content != original:
        with open(file_path, 'w') as f:
            f.write(content)
        print(f"  Sanitized: {file_path.name} ({n_changes} path references)")

    return n_changes


def update_file_references(qa_reports_dir: Path, rename_map: Dict[Path, Path], dry_run: bool) -> int:
    """Update references to renamed files in QA reports."""
    total_updates = 0

    for json_file in qa_reports_dir.glob("*.json"):
        with open(json_file, 'r') as f:
            content = f.read()

        original = content

        # Replace old filenames with new ones
        for old_path, new_path in rename_map.items():
            content = content.replace(old_path.name, new_path.name)

        if content != original:
            n_updates = sum(1 for old_path in rename_map if old_path.name in original)
            total_updates += n_updates

            if not dry_run:
                with open(json_file, 'w') as f:
                    f.write(content)
                print(f"  Updated references in: {json_file.name}")

    return total_updates


def generate_git_instructions(repo_root: Path) -> str:
    """Generate instructions for creating anonymous git history."""
    return f"""
================================================================================
GIT ANONYMIZATION INSTRUCTIONS
================================================================================

The current git history contains your identity. To create an anonymous repo:

OPTION 1: Create fresh repo (RECOMMENDED)
-----------------------------------------
cd {repo_root}
rm -rf .git
git init
git add .
git commit -m "Initial release: CEI Benchmark dataset v1.0" --author="Anonymous <anonymous@example.org>"

Then push to an anonymous hosting service like:
- https://anonymous.4open.science/
- https://osf.io/ (with anonymous view link)

OPTION 2: Rewrite history (if you must keep the same repo)
----------------------------------------------------------
# WARNING: This will change all commit hashes
git filter-branch --env-filter '
export GIT_AUTHOR_NAME="Anonymous"
export GIT_AUTHOR_EMAIL="anonymous@example.org"
export GIT_COMMITTER_NAME="Anonymous"
export GIT_COMMITTER_EMAIL="anonymous@example.org"
' --tag-name-filter cat -- --branches --tags

git push --force --all
git push --force --tags

NOTE: GitHub may cache the old commits. A fresh repo is safer for blind review.

OPTION 3: Use BFG Repo-Cleaner (alternative to filter-branch)
------------------------------------------------------------
# Install: brew install bfg
bfg --replace-text passwords.txt  # if you need to scrub specific strings
================================================================================
"""


def main():
    parser = argparse.ArgumentParser(description="Anonymize CEI dataset for blind review")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without applying")
    args = parser.parse_args()

    repo_root = get_repo_root()
    gold_dir = repo_root / "data" / "gold"
    qa_reports_dir = repo_root / "data" / "qa_reports"

    print("=" * 70)
    print("CEI DATASET ANONYMIZATION SCRIPT")
    print("=" * 70)

    if args.dry_run:
        print("\n*** DRY RUN MODE - No changes will be made ***\n")

    # Step 1: Rename gold files
    print("\n[1/4] Renaming gold data files to remove annotator names...")
    rename_map = build_rename_map(gold_dir)

    if args.dry_run:
        print("\n  Planned renames:")
        for old_path, new_path in rename_map.items():
            print(f"    {old_path.name}")
            print(f"      -> {new_path.name}")
    else:
        rename_gold_files(rename_map, dry_run=False)

    # Step 2: Sanitize QA reports (remove local paths)
    print("\n[2/4] Sanitizing QA reports to remove local paths...")

    if qa_reports_dir.exists():
        for json_file in qa_reports_dir.glob("*.json"):
            n_changes = sanitize_json_paths(json_file, args.dry_run)
            if args.dry_run and n_changes > 0:
                print(f"    Would sanitize: {json_file.name} ({n_changes} paths)")

    # Step 3: Update file references in QA reports
    print("\n[3/4] Updating file references in QA reports...")

    if qa_reports_dir.exists() and not args.dry_run:
        # Need to rebuild rename_map with new paths for reference updates
        update_file_references(qa_reports_dir, rename_map, args.dry_run)
    elif args.dry_run:
        print("    Would update filename references in all QA report JSON files")

    # Step 4: Check for remaining PII
    print("\n[4/4] Checking for remaining PII patterns...")

    pii_patterns = [
        (r'jonc', "Username 'jonc'"),
        (r'jon-chun', "GitHub username"),
        (r'jonchun@', "Email address"),
        (r'/Users/[a-zA-Z]+/', "Local user path"),
    ]

    remaining_issues = []
    for pattern, description in pii_patterns:
        for file_path in repo_root.rglob("*"):
            if file_path.is_file() and ".git" not in str(file_path):
                try:
                    content = file_path.read_text(errors='ignore')
                    if re.search(pattern, content, re.IGNORECASE):
                        remaining_issues.append((file_path.relative_to(repo_root), description))
                except:
                    pass

    if remaining_issues:
        print("\n  WARNING: Remaining PII found:")
        for file_path, description in set(remaining_issues):
            print(f"    - {file_path}: {description}")
    else:
        print("    No remaining PII patterns found in source files.")

    # Git instructions
    print(generate_git_instructions(repo_root))

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Gold files to rename: {len(rename_map)}")
    print(f"  QA reports to sanitize: {len(list(qa_reports_dir.glob('*.json'))) if qa_reports_dir.exists() else 0}")

    if args.dry_run:
        print("\n  Run without --dry-run to apply changes.")
    else:
        print("\n  Changes applied successfully!")
        print("  IMPORTANT: You still need to create a fresh git history (see instructions above)")


if __name__ == "__main__":
    main()
