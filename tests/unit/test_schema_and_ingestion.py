import json

from src.schema import DbSchema, SchemaCatalog, quote_ident


def test_ingestion_outputs(data_dir):
    processed = data_dir / "processed"
    train = [json.loads(line) for line in (processed / "train.jsonl").open(encoding="utf-8")]
    dev = [json.loads(line) for line in (processed / "dev.jsonl").open(encoding="utf-8")]
    assert len(train) == 7  # train_spider.json + train_others.json
    assert len(dev) == 4
    assert set(train[0]) == {"db_id", "question", "query", "schema_sql"}
    assert "PRIMARY KEY" in dev[0]["schema_sql"] and "FOREIGN KEY" in dev[0]["schema_sql"]
    assert set(json.loads((processed / "schemas.json").read_text())) == {"concert_singer", "pets_1"}


def test_from_spider_maps_keys(data_dir):
    entry = next(
        e
        for e in json.loads((data_dir / "raw/spider_data/tables.json").read_text())
        if e["db_id"] == "concert_singer"
    )
    schema = DbSchema.from_spider(entry)
    assert schema.table_names == ["stadium", "singer", "concert", "singer_in_concert"]
    assert schema.table("SINGER").primary_key == ["Singer_ID"]
    assert schema.table("singer").columns[3].type == "NUMBER"
    fk = {(f.table, f.column, f.ref_table, f.ref_column) for f in schema.foreign_keys}
    assert ("concert", "Stadium_ID", "stadium", "Stadium_ID") in fk
    assert schema.neighbors("singer") == {"singer_in_concert"}


def test_create_sql_and_subset(data_dir):
    catalog = SchemaCatalog.from_json(data_dir / "processed/schemas.json")
    schema = catalog.get("concert_singer")
    full = schema.to_create_sql()
    assert full.count("CREATE TABLE") == 4
    assert "FOREIGN KEY (Stadium_ID) REFERENCES stadium(Stadium_ID)" in full
    partial = schema.to_create_sql(["concert", "singer_in_concert"])
    assert partial.count("CREATE TABLE") == 2
    assert "REFERENCES stadium" not in partial  # FK to a pruned table is dropped


def test_from_sqlite_introspection(data_dir):
    path = data_dir / "raw/spider_data/database/pets_1/pets_1.sqlite"
    schema = DbSchema.from_sqlite(path)
    assert schema.db_id == "pets_1"
    assert schema.table("student").primary_key == ["StuID"]
    assert {(f.table, f.ref_table) for f in schema.foreign_keys} == {
        ("has_pet", "student"),
        ("has_pet", "pets"),
    }


def test_catalog_roundtrip_and_paths(data_dir, tmp_path):
    catalog = SchemaCatalog.from_json(
        data_dir / "processed/schemas.json", [data_dir / "raw/spider_data/database"]
    )
    out = tmp_path / "schemas.json"
    catalog.save_json(out)
    again = SchemaCatalog.from_json(out)
    assert again.get("pets_1").to_create_sql() == catalog.get("pets_1").to_create_sql()
    assert catalog.db_path("pets_1").name == "pets_1.sqlite"
    assert catalog.db_path("missing") is None


def test_quote_ident():
    assert quote_ident("singer") == "singer"
    assert quote_ident("Song Name") == '"Song Name"'
    assert quote_ident('we"ird') == '"we""ird"'
