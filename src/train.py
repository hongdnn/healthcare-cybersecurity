from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, Dict

import pandas as pd


READERS: Dict[str, Callable[[Path], pd.DataFrame]] = {
    ".csv": lambda p: pd.read_csv(p),
    ".tsv": lambda p: pd.read_csv(p, sep="\t"),
    ".parquet": lambda p: pd.read_parquet(p),
    ".xlsx": lambda p: pd.read_excel(p),
    ".xls": lambda p: pd.read_excel(p),
}


def load_file(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix not in READERS:
        raise ValueError(f"Unsupported file type: {path.name}")
    return READERS[suffix](path)


def load_datasets(data_dir: Path, expected_files: list[str] | None) -> Dict[str, pd.DataFrame]:
    if expected_files:
        paths = [data_dir / name for name in expected_files]
    else:
        paths = [p for p in data_dir.iterdir() if p.is_file()]
        if len(paths) != 5:
            raise ValueError(
                f"Expected 5 dataset files in {data_dir}, found {len(paths)}. "
                "Use --expected to provide exact filenames."
            )

    missing = [p.name for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing dataset files: {', '.join(missing)}")

    datasets: Dict[str, pd.DataFrame] = {}
    for path in sorted(paths):
        df = load_file(path)
        datasets[path.stem] = df
        print(f"Loaded {path.name}: {df.shape[0]} rows, {df.shape[1]} columns")
        print(df.head())
        print("-" * 80)

    return datasets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load CICIoMT2024 datasets")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "datasets",
        help="Path to datasets directory",
    )
    parser.add_argument(
        "--expected",
        type=str,
        default="",
        help="Comma-separated dataset filenames to load",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    expected = [x.strip() for x in args.expected.split(",") if x.strip()]
    load_datasets(args.data_dir, expected if expected else None)


if __name__ == "__main__":
    main()
