#!/usr/bin/env python3
"""Download images from URLs listed in a CSV column (fixtureImage)."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

COLUMN = "fixtureImage"
DEFAULT_CSV = "Fixture Images - Sheet1.csv"
DEFAULT_OUT = "downloaded_images"
USER_AGENT = "download_images_from_csv/1.0"


def unique_path(directory: Path, basename: str) -> Path:
    """Return path/dir/basename, or dir/name_1.ext if basename exists."""
    path = directory / basename
    if not path.exists():
        return path
    stem = Path(basename).stem
    suffix = Path(basename).suffix
    n = 1
    while True:
        candidate = directory / f"{stem}_{n}{suffix}"
        if not candidate.exists():
            return candidate
        n += 1


def download_one(url: str, dest: Path) -> None:
    req = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(req, timeout=60) as resp:
        dest.write_bytes(resp.read())


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download images from URLs in a CSV column named fixtureImage."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path(DEFAULT_CSV),
        help=f"Path to CSV file (default: {DEFAULT_CSV} in current working directory)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(DEFAULT_OUT),
        help=f"Output directory (default: {DEFAULT_OUT})",
    )
    args = parser.parse_args()

    csv_path = args.csv.resolve()
    if not csv_path.is_file():
        print(f"Error: CSV not found: {csv_path}", file=sys.stderr)
        return 1

    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    ok = skipped = failed = 0
    failures: list[str] = []

    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or COLUMN not in reader.fieldnames:
            print(
                f"Error: CSV must have a '{COLUMN}' column. Found: {reader.fieldnames}",
                file=sys.stderr,
            )
            return 1

        for row in reader:
            raw = (row.get(COLUMN) or "").strip()
            if not raw:
                skipped += 1
                continue

            parsed = urlparse(raw)
            if parsed.scheme not in ("http", "https") or not parsed.netloc:
                print(f"Skip invalid URL: {raw!r}", file=sys.stderr)
                failed += 1
                failures.append(raw)
                continue

            segment = Path(parsed.path).name
            if not segment or segment in (".", ".."):
                segment = "image"
                if not Path(segment).suffix:
                    segment += ".bin"

            dest = unique_path(out_dir, segment)
            try:
                download_one(raw, dest)
                ok += 1
            except (HTTPError, URLError, OSError, TimeoutError, ValueError) as e:
                print(f"Failed {raw}: {e}", file=sys.stderr)
                failed += 1
                failures.append(raw)

    print(f"Done: {ok} downloaded, {skipped} empty rows skipped, {failed} failed.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
