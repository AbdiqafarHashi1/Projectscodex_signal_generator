#!/usr/bin/env python3
"""Extract a strategy update archive produced by create_update_archive.py."""
from __future__ import annotations

import argparse
import base64
import io
from pathlib import Path
from zipfile import ZipFile


def _load_archive_bytes(path: Path, *, treat_as_base64: bool | None) -> bytes:
    if treat_as_base64 is None:
        treat_as_base64 = path.suffix == ".b64"

    if treat_as_base64:
        data = path.read_text()
        return base64.b64decode(data)

    return path.read_bytes()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help="Path to the .zip or .zip.b64 archive")
    parser.add_argument(
        "--dest",
        type=Path,
        default=Path("delivery/latest_strategy_update"),
        help="Directory to extract files into (default: delivery/latest_strategy_update)",
    )
    parser.add_argument(
        "--base64",
        action="store_true",
        help="Force treating the archive as base64 even if the suffix is not .b64",
    )
    parser.add_argument(
        "--binary",
        action="store_true",
        help="Force treating the archive as binary even if the suffix is .b64",
    )
    args = parser.parse_args()

    if args.base64 and args.binary:
        raise SystemExit("Choose only one of --base64 or --binary")

    base64_hint: bool | None
    if args.base64:
        base64_hint = True
    elif args.binary:
        base64_hint = False
    else:
        base64_hint = None

    archive_bytes = _load_archive_bytes(args.archive, treat_as_base64=base64_hint)

    args.dest.mkdir(parents=True, exist_ok=True)
    with ZipFile(io.BytesIO(archive_bytes)) as zip_file:
        zip_file.extractall(args.dest)

    rel_archive = args.archive.resolve()
    rel_dest = args.dest.resolve()
    print(f"Extracted {rel_archive} into {rel_dest}")


if __name__ == "__main__":
    main()
