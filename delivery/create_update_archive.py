#!/usr/bin/env python3
"""Create a transport-friendly archive of the tracked strategy files."""
from __future__ import annotations

import argparse
import base64
import io
import subprocess
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


def _list_tracked_files(repo_root: Path) -> list[Path]:
    """Return all git-tracked files relative to *repo_root*."""
    result = subprocess.run(
        ["git", "ls-files"], check=True, capture_output=True, text=True, cwd=repo_root
    )
    return [repo_root / line.strip() for line in result.stdout.splitlines() if line.strip()]


def _build_zip_bytes(repo_root: Path, tracked_files: list[Path]) -> bytes:
    buffer = io.BytesIO()
    with ZipFile(buffer, "w", compression=ZIP_DEFLATED) as zip_file:
        for path in tracked_files:
            if not path.exists():
                # Skip files removed locally but still tracked in history.
                continue
            arcname = path.relative_to(repo_root)
            zip_file.write(path, arcname.as_posix())
    return buffer.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--format",
        choices={"zip", "base64"},
        default="zip",
        help=(
            "Set to 'base64' to emit a text archive suitable for environments that "
            "reject binary uploads."
        ),
    )
    parser.add_argument(
        "--output",
        help="Optional output filename. Defaults to delivery/latest_strategy_update.<ext>",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    delivery_dir = Path(__file__).resolve().parent
    tracked_files = _list_tracked_files(repo_root)
    if not tracked_files:
        raise SystemExit("No tracked files found to package")

    zip_bytes = _build_zip_bytes(repo_root, tracked_files)

    if args.format == "zip":
        archive_path = (
            Path(args.output)
            if args.output
            else delivery_dir / "latest_strategy_update.zip"
        )
        archive_path.write_bytes(zip_bytes)
    else:
        archive_path = (
            Path(args.output)
            if args.output
            else delivery_dir / "latest_strategy_update.zip.b64"
        )
        base64_data = base64.b64encode(zip_bytes).decode("ascii")
        archive_path.write_text(base64_data)

    rel_path = archive_path.relative_to(repo_root)
    print(f"Created {rel_path} with {len(tracked_files)} files ({args.format})")


if __name__ == "__main__":
    main()
