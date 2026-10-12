"""
Spider dataset download and preprocessing.

Downloads the Spider Text-to-SQL benchmark and prepares:
  - data/processed/{train,dev,test}.jsonl : question / gold SQL / schema records
  - data/processed/schemas.json           : structured schema for every database

Usage:
    python -m src.ingestion.download_spider
    python -m src.ingestion.download_spider --raw-dir data/raw --processed-dir data/processed
"""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

from src.schema import DbSchema, SchemaCatalog

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

# Google Drive file id of the official spider_data.zip (2024 release incl. the test set),
# linked from https://yale-lily.github.io/spider
SPIDER_GDRIVE_ID = "1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J"
ZIP_NAMES = ("spider_data.zip", "spider.zip")
EXTRACTED_NAMES = ("spider_data", "spider")  # 2024 release / original 2018 release

# Spider ships train as two files; dev/test are single files
SPLIT_FILES = {
    "train": ["train_spider.json", "train_others.json"],
    "dev": ["dev.json"],
    "test": ["test.json"],
}


def find_spider_dir(raw_dir: Path = RAW_DIR) -> Path | None:
    for name in EXTRACTED_NAMES:
        candidate = raw_dir / name
        if (candidate / "tables.json").exists():
            return candidate
    return None


def download_spider(raw_dir: Path = RAW_DIR) -> Path | None:
    """Download spider_data.zip with gdown if it isn't already present."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    for name in ZIP_NAMES:
        if (raw_dir / name).exists():
            print(f"Spider zip already downloaded: {raw_dir / name}")
            return raw_dir / name
    zip_path = raw_dir / ZIP_NAMES[0]
    try:
        import gdown  # optional dependency (requirements-dev.txt)
    except ImportError:
        print("gdown is not installed. Either `pip install gdown` and re-run, or download")
        print("spider_data.zip manually from https://yale-lily.github.io/spider")
        print(f"and place it at {zip_path}.")
        return None
    print(f"Downloading Spider dataset to {zip_path} ...")
    gdown.download(id=SPIDER_GDRIVE_ID, output=str(zip_path), quiet=False)
    return zip_path if zip_path.exists() else None


def extract_spider(raw_dir: Path = RAW_DIR) -> Path:
    """Extract the Spider zip (skipping macOS metadata) and return the dataset dir."""
    zip_path = next((raw_dir / n for n in ZIP_NAMES if (raw_dir / n).exists()), None)
    if zip_path is None:
        raise FileNotFoundError(f"No Spider zip in {raw_dir}. Run download_spider() first.")
    with zipfile.ZipFile(zip_path) as zf:
        members = [
            n for n in zf.namelist() if not n.startswith("__MACOSX") and not n.endswith(".DS_Store")
        ]
        zf.extractall(raw_dir, members=members)
    spider_dir = find_spider_dir(raw_dir)
    if spider_dir is None:
        raise FileNotFoundError(f"Extracted {zip_path} but found no tables.json under {raw_dir}")
    print(f"Extracted Spider to {spider_dir}")
    return spider_dir


def load_schemas(spider_dir: Path) -> dict[str, dict]:
    """Load raw schema entries from tables.json (+ test_tables.json when present).

    Returns {db_id: tables.json entry}.
    """
    schemas: dict[str, dict] = {}
    for fname in ("tables.json", "test_tables.json"):
        path = spider_dir / fname
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as f:
            for db in json.load(f):
                schemas.setdefault(db["db_id"], db)
    return schemas


def schema_to_create_statements(db_schema: dict) -> str:
    """Convert a Spider schema entry to CREATE TABLE statements with PK/FK constraints."""
    return DbSchema.from_spider(db_schema).to_create_sql()


def process_split(split: str, spider_dir: Path, schemas: dict[str, dict]) -> list[dict]:
    """
    Process a train/dev/test split into structured records.

    Returns list of dicts with keys: db_id, question, query, schema_sql
    """
    rendered: dict[str, str] = {}
    records = []
    for fname in SPLIT_FILES[split]:
        path = spider_dir / fname
        if not path.exists():
            raise FileNotFoundError(f"{path} not found (split '{split}')")
        with open(path, encoding="utf-8") as f:
            examples = json.load(f)
        for ex in examples:
            db_id = ex["db_id"]
            if db_id not in rendered:
                rendered[db_id] = schema_to_create_statements(schemas[db_id])
            records.append(
                {
                    "db_id": db_id,
                    "question": ex["question"].strip(),
                    "query": ex["query"].strip(),
                    "schema_sql": rendered[db_id],
                }
            )
    return records


def save_processed(records: list[dict], split: str, processed_dir: Path = PROCESSED_DIR) -> Path:
    """Save processed records to JSONL."""
    processed_dir.mkdir(parents=True, exist_ok=True)
    out_path = processed_dir / f"{split}.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Saved {len(records)} {split} records to {out_path}")
    return out_path


def build_catalog(spider_dir: Path, schemas: dict[str, dict]) -> SchemaCatalog:
    db_dirs = [spider_dir / "database", spider_dir / "test_database"]
    return SchemaCatalog({k: DbSchema.from_spider(v) for k, v in schemas.items()}, db_dirs)


def run(raw_dir: Path = RAW_DIR, processed_dir: Path = PROCESSED_DIR) -> None:
    spider_dir = find_spider_dir(raw_dir)
    if spider_dir is None:
        if download_spider(raw_dir) is None:
            return
        spider_dir = extract_spider(raw_dir)

    schemas = load_schemas(spider_dir)
    catalog = build_catalog(spider_dir, schemas)
    catalog.save_json(processed_dir / "schemas.json")
    print(f"Saved {len(catalog)} database schemas to {processed_dir / 'schemas.json'}")

    for split in SPLIT_FILES:
        if not all((spider_dir / f).exists() for f in SPLIT_FILES[split]):
            print(f"Skipping '{split}': files not present in {spider_dir}")
            continue
        save_processed(process_split(split, spider_dir, schemas), split, processed_dir)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download and preprocess Spider")
    p.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    p.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.raw_dir, args.processed_dir)
