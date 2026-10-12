import pytest

from src.retrieval.context import ContextBuilder
from src.retrieval.example_retriever import Example, ExampleRetriever, question_skeleton
from src.retrieval.schema_retriever import SchemaRetriever
from src.retrieval.text import split_identifier, stem, tokenize
from src.retrieval.value_index import ValueIndex
from src.schema import SchemaCatalog


@pytest.fixture(scope="module")
def catalog(data_dir):
    return SchemaCatalog.from_json(
        data_dir / "processed/schemas.json", [data_dir / "raw/spider_data/database"]
    )


@pytest.fixture(scope="module")
def retriever(catalog):
    return SchemaRetriever.from_catalog(catalog)


def test_tokenization():
    assert split_identifier("Song_release_year") == ["song", "release", "year"]
    assert split_identifier("concertName") == ["concert", "name"]
    assert (
        stem("singers") == "singer"
        and stem("countries") == "country"
        and stem("classes") == "class"
    )
    assert tokenize("How many singers are there?") == ["singer"]


def test_database_routing(retriever):
    assert retriever.route("Which students own a dog?") == "pets_1"
    assert retriever.route("List the singers and their concerts") == "concert_singer"
    full = retriever.retrieve("anything", db_id="pets_1")
    assert full[0]["score"] == 1.0 and "CREATE TABLE student" in full[0]["schema_sql"]


def test_table_pruning_keeps_fk_neighbours(retriever):
    ranked = retriever.rank_tables("What is the capacity of each stadium?", "concert_singer")
    assert ranked[0][0] == "stadium"
    selected = retriever.select_tables(
        "What is the capacity of each stadium?", "concert_singer", max_tables=1
    )
    assert selected == ["stadium", "concert"]  # concert joins to stadium through a foreign key
    no_fk = retriever.select_tables(
        "capacity of each stadium", "concert_singer", 1, expand_foreign_keys=False
    )
    assert no_fk == ["stadium"]


def test_small_database_is_not_pruned(retriever):
    assert len(retriever.select_tables("stadium", "concert_singer", max_tables=6)) == 4


def test_no_lexical_signal_keeps_everything(retriever):
    assert len(retriever.select_tables("zzz qqq", "concert_singer", max_tables=1)) == 4


def test_example_retrieval_is_structural(catalog):
    examples = [
        Example("How many pets are there?", "SELECT count(*) FROM pets", "pets_1"),
        Example("What is the average age of students?", "SELECT avg(Age) FROM student", "pets_1"),
        Example("How many students are there?", "SELECT count(*) FROM student", "pets_1"),
        Example("How many cars do we have?", "SELECT count(*) FROM cars", "car_1"),
    ]
    retriever = ExampleRetriever(examples, catalog)
    top = retriever.retrieve("How many singers do we have?", k=2, db_id="concert_singer")
    assert top[0].question == "How many cars do we have?"
    assert len({t.query.lower() for t in top}) == 2  # distinct SQL templates
    assert all(
        e.db_id != "car_1"
        for e in retriever.retrieve("How many singers?", 3, exclude_db_id="car_1")
    )
    assert retriever.retrieve("anything", k=0) == []


def test_question_skeleton_masks_schema_terms(catalog):
    skeleton = question_skeleton(
        "How many singers are older than 30?", catalog.get("concert_singer")
    )
    assert "singer" not in skeleton and "30" not in skeleton
    assert "how" in skeleton and "many" in skeleton


def test_value_hints(catalog):
    index = ValueIndex(catalog)
    hints = index.match("Which singers are from France?", "concert_singer")
    assert [h.render() for h in hints] == ["singer.Country = 'France'"]
    assert index.match("Which singers are from France?", "concert_singer", tables=["stadium"]) == []
    assert index.match("concerts in 2014", "concert_singer")[0].render() == "concert.Year = '2014'"
    quoted = index.match('stadium named "Stark\'s Park"', "concert_singer")
    assert quoted and quoted[0].value == "Stark's Park" and "''" in quoted[0].render()


def test_context_builder_modes(catalog, retriever, data_dir):
    builder = ContextBuilder(
        catalog,
        retriever,
        ExampleRetriever.from_jsonl(data_dir / "processed" / "train.jsonl", catalog),
        ValueIndex(catalog),
        num_examples=2,
        max_tables=1,
    )
    full = builder.build("qlora", "Which singers are from France?", "concert_singer")
    assert full.schema_sql.count("CREATE TABLE") == 4 and not full.examples and not full.value_hints

    rag = builder.build("rag", "Which singers are from France?", "concert_singer")
    assert "singer" in rag.tables and len(rag.examples) == 2
    assert rag.value_hints == ["singer.Country = 'France'"]

    forced = builder.build(
        "rag", "Which singers are from France?", "concert_singer", must_include_tables=["stadium"]
    )
    assert "stadium" in forced.tables

    no_self = builder.build(
        "rag", "How many pets are there?", "pets_1", exclude_example_db="pets_1"
    )
    assert no_self.examples == []  # the mini train set only has pets_1 examples


def test_bm25_small_corpus_is_not_degenerate():
    """Robertson IDF is 0 for a term in 1 of 2 docs; the Lucene IDF must still rank correctly."""
    from src.retrieval.bm25 import BM25

    bm25 = BM25([["singer", "concert"], ["student", "pet", "dog"]])
    scores = bm25.get_scores(["dog"])
    assert scores[1] > 0 and scores[0] == 0
    assert BM25([["a"], ["b"]]).get_scores(["zzz"]).tolist() == [0.0, 0.0]
