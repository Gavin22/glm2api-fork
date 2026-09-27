#!/usr/bin/env python3
"""Merge tracked files from a glm2api git bundle into a deployed checkout.

Updates only files git already tracks, so deployment-private files such as
`.env` are never touched. Writes what it does to a manifest so the update can
be rolled back file by file.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if result.returncode != 0:
        raise SystemExit(f"git {' '.join(args)} failed:\n{result.stderr.strip()}")
    return result.stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--settle", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    bundle = Path(args.bundle).resolve()
    settle = Path(args.settle).resolve()
    manifest_path = Path(args.manifest).resolve()

    if not repo.is_dir():
        raise SystemExit(f"repo directory not found: {repo}")

    settle.mkdir(parents=True, exist_ok=True)

    # Fetching the bundle adds one ref; it does not touch the working tree.
    ref = "refs/bundle-update"
    git(repo, "fetch", "--force", str(bundle), f"master:{ref}")
    tracked = [line for line in git(repo, "ls-tree", "-r", "--name-only", ref).splitlines() if line]

    updated: list[str] = []
    for relative in tracked:
        blob = subprocess.run(
            ["git", "-C", str(repo), "show", f"{ref}:{relative}"],
            capture_output=True,
        )
        if blob.returncode != 0:
            raise SystemExit(f"cannot read {relative} from {ref}")
        target = settle / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(blob.stdout)
        updated.append(relative)

    manifest_path.write_text(
        json.dumps({"ref": ref, "files": updated}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"{ref} -> {settle}")
    print(f"{len(updated)} tracked files written; .env and untracked files untouched")
    return 0


if __name__ == "__main__":
    sys.exit(main())
