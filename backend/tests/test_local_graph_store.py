"""Local graph store tests against a real SQLite file (the LLM is a stub)."""

import json
import sqlite3
import time

import pytest

from app.config import Config
from app.services import graph_builder as graph_builder_module
from app.services.graph_builder import GraphBuilderService
from app.services.zep_entity_reader import ZepEntityReader
from app.utils import zep
from app.utils.local_graph import (
    BatchAddItem,
    GraphExtractionError,
    LocalGraphClient,
    NotFoundError,
)
from app.utils.zep_paging import fetch_all_edges, fetch_all_nodes

ONTOLOGY = {
    "entity_types": [
        {"name": "Person", "description": "A person", "attributes": [{"name": "role", "description": "role"}]},
        {"name": "Company", "description": "A company", "attributes": [{"name": "details", "description": "details"}]},
    ],
    "edge_types": [
        {
            "name": "WORKS_AT",
            "description": "employment",
            "attributes": [],
            "source_targets": [{"source": "Person", "target": "Company"}],
        }
    ],
}

# Fixed extraction per marker word found in the text.
EXTRACTIONS = {
    "alpha": {
        "entities": [
            {"name": "Alice", "type": "Person", "summary": "Alice is an engineer.", "attributes": {"role": "engineer"}},
            {"name": "Acme", "type": "Company", "summary": "Acme builds rockets.", "attributes": {}},
        ],
        "relations": [
            {"source": "Alice", "target": "Acme", "type": "WORKS_AT", "fact": "Alice works at Acme"},
        ],
    },
    "beta": {
        "entities": [
            {"name": "alice ", "type": "Person", "summary": "Alice lives in Paris.", "attributes": {}},
            {"name": "Bob", "type": "Unknown", "summary": "Bob is a neighbour.", "attributes": {}},
        ],
        "relations": [
            {"source": "Alice", "target": "Acme", "type": "WORKS_AT", "fact": "Alice works at Acme"},
            {"source": "Bob", "target": "Alice", "type": "KNOWS", "fact": "Bob knows Alice"},
            {"source": "Bob", "target": "\u5f20\u4e09", "type": "KNOWS", "fact": "Bob knows \u5f20\u4e09"},
        ],
    },
}


class StubLLM:
    """Returns fixed JSON; records each call. Not a database mock."""

    def __init__(self, fail=False, payload=None):
        self.fail = fail
        self.payload = payload
        self.calls = []

    def chat_json(self, messages, temperature=0.3, max_tokens=4096, max_attempts=1):
        self.calls.append(messages)
        if self.fail:
            raise RuntimeError("provider unavailable")
        if self.payload is not None:
            return self.payload
        text = messages[-1]["content"]
        result = {"entities": [], "relations": []}
        for marker, extraction in EXTRACTIONS.items():
            if marker in text:
                result["entities"] += extraction["entities"]
                result["relations"] += extraction["relations"]
        return result


def _client(tmp_path, llm=None, name="graph.sqlite3"):
    return LocalGraphClient(str(tmp_path / name), llm=llm or StubLLM())


def _new_graph(client, graph_id="graph-a"):
    client.graph.create(graph_id=graph_id, name=graph_id)
    client.graph.set_ontology(graph_ids=[graph_id], ontology=ONTOLOGY)
    return graph_id


def _wait(client, batch_id, timeout=10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        batch = client.batch.get(batch_id=batch_id)
        if batch.status in {"succeeded", "failed"}:
            return batch
        time.sleep(0.05)
    raise AssertionError("batch did not finish")


def _ingest(client, graph_id, texts):
    batch = client.batch.create(metadata={"graph_id": graph_id})
    client.batch.add(
        batch_id=batch.batch_id,
        items=[BatchAddItem(type="graph_episode", graph_id=graph_id, data=text) for text in texts],
    )
    client.batch.process(batch_id=batch.batch_id)
    return _wait(client, batch.batch_id)


def _all(client, graph_id):
    nodes = fetch_all_nodes(client, graph_id)
    edges = fetch_all_edges(client, graph_id)
    return nodes, edges


def test_add_process_creates_deduplicated_nodes_and_edges(tmp_path):
    llm = StubLLM()
    client = _client(tmp_path, llm)
    graph_id = _new_graph(client)

    batch = _ingest(client, graph_id, ["alpha text", "beta text"])

    assert batch.status == "succeeded"
    assert batch.progress.succeeded_items == 2
    assert batch.progress.percent_complete == 100.0
    # Both chunks fit one extraction call.
    assert len(llm.calls) == 1
    assert "WORKS_AT" in llm.calls[0][0]["content"]  # ontology given to the LLM

    nodes, edges = _all(client, graph_id)
    by_name = {node.name: node for node in nodes}
    assert sorted(by_name) == ["Acme", "Alice", "Bob", "\u5f20\u4e09"]
    alice = by_name["Alice"]
    assert alice.labels == ["Entity", "Person"]
    assert alice.summary == "Alice is an engineer. Alice lives in Paris."
    assert alice.attributes == {"role": "engineer"}
    assert by_name["Bob"].labels == ["Entity"]  # type outside the ontology

    facts = sorted((edge.name, edge.fact) for edge in edges)
    assert facts == [
        ("RELATES_TO", "Bob knows Alice"),
        ("RELATES_TO", "Bob knows \u5f20\u4e09"),
        ("WORKS_AT", "Alice works at Acme"),
    ]
    works_at = next(edge for edge in edges if edge.name == "WORKS_AT")
    assert works_at.source_node_uuid == alice.uuid_
    assert works_at.target_node_uuid == by_name["Acme"].uuid_
    assert len(works_at.episodes) == 2

    items = client.batch.list_items(batch_id=batch.batch_id).items
    for item in items:
        episode = client.graph.episode.get(uuid_=item.episode_uuid, graph_id=graph_id)
        assert episode.processed is True

    # Re-ingesting the same facts does not duplicate nodes or edges.
    _ingest(client, graph_id, ["alpha again"])
    nodes_again, edges_again = _all(client, graph_id)
    assert len(nodes_again) == len(nodes)
    assert len(edges_again) == len(edges)

    # The entity reader keeps only ontology-typed entities.
    filtered = ZepEntityReader.__new__(ZepEntityReader)
    filtered.client = client
    result = filtered.filter_defined_entities(graph_id)
    assert sorted(entity.name for entity in result.entities) == ["Acme", "Alice"]
    acme = next(entity for entity in result.entities if entity.name == "Acme")
    assert [edge["direction"] for edge in acme.related_edges] == ["incoming"]
    context = filtered.get_entity_with_context(graph_id, alice.uuid_)
    assert {edge["direction"] for edge in context.related_edges} == {"incoming", "outgoing"}


def test_graph_add_extracts_synchronously(tmp_path):
    client = _client(tmp_path)
    graph_id = _new_graph(client)

    episode = client.graph.add(
        graph_id=graph_id, type="text", data="alpha activity",
        created_at="2026-07-22T12:00:00+08:00", source_description="sim",
    )

    assert episode.processed is True
    _, edges = _all(client, graph_id)
    assert [edge.fact for edge in edges] == ["Alice works at Acme"]
    assert edges[0].valid_at == "2026-07-22T12:00:00+08:00"


def test_search_honours_scope_and_limit(tmp_path):
    client = _client(tmp_path)
    graph_id = _new_graph(client)
    _ingest(client, graph_id, ["alpha", "beta"])

    edges = client.graph.search(graph_id=graph_id, query="who works at Acme", limit=5, scope="edges", reranker="rrf")
    assert edges.nodes == []
    assert edges.edges[0].fact == "Alice works at Acme"

    nodes = client.graph.search(graph_id=graph_id, query="rockets", limit=5, scope="nodes")
    assert nodes.edges == []
    assert [node.name for node in nodes.nodes] == ["Acme"]

    limited = client.graph.search(graph_id=graph_id, query="Alice Bob", limit=1, scope="edges")
    assert len(limited.edges) == 1

    # Two-character CJK names are below the trigram size: substring fallback.
    # "\u5f20\u4e09" is a two-character CJK name (Zhang San), written as escapes.
    short = client.graph.search(graph_id=graph_id, query="\u5f20\u4e09", limit=5, scope="edges")
    assert [edge.fact for edge in short.edges] == ["Bob knows \u5f20\u4e09"]

    assert client.graph.search(graph_id=graph_id, query="nothing matches", limit=5, scope="edges").edges == []


def test_delete_removes_everything(tmp_path):
    client = _client(tmp_path)
    graph_id = _new_graph(client)
    batch = _ingest(client, graph_id, ["alpha", "beta"])

    client.graph.delete(graph_id)

    with pytest.raises(NotFoundError):
        client.graph.get(graph_id)
    with pytest.raises(NotFoundError):
        client.graph.delete(graph_id)
    with pytest.raises(NotFoundError):
        client.batch.get(batch_id=batch.batch_id)
    connection = sqlite3.connect(client.store.db_path)
    try:
        for table in ("nodes", "edges", "episodes", "batch_items", "nodes_fts", "edges_fts"):
            count = connection.execute(
                f"SELECT COUNT(*) FROM {table} WHERE graph_id = ?", (graph_id,)
            ).fetchone()[0]
            assert count == 0, table
    finally:
        connection.close()


def test_graphs_are_isolated_through_every_read_method(tmp_path):
    client = _client(tmp_path)
    graph_a = _new_graph(client, "graph-a")
    graph_b = _new_graph(client, "graph-b")
    batch_a = _ingest(client, graph_a, ["alpha", "beta"])

    nodes_a, edges_a = _all(client, graph_a)
    node_a = nodes_a[0]
    episode_a = client.batch.list_items(batch_id=batch_a.batch_id).items[0].episode_uuid

    assert _all(client, graph_b) == ([], [])
    assert client.graph.node.get_by_graph_id(graph_b) == []
    assert client.graph.edge.get_by_graph_id(graph_b) == []
    with pytest.raises(NotFoundError):
        client.graph.node.get(uuid_=node_a.uuid_, graph_id=graph_b)
    with pytest.raises(NotFoundError):
        client.graph.node.get_edges(node_uuid=node_a.uuid_, graph_id=graph_b)
    with pytest.raises(NotFoundError):
        client.graph.episode.get(uuid_=episode_a, graph_id=graph_b)
    for scope in ("edges", "nodes"):
        result = client.graph.search(graph_id=graph_b, query="Alice Acme \u5f20\u4e09", limit=10, scope=scope)
        assert result.edges == [] and result.nodes == []

    # Same entity name in graph B becomes its own node; graph A is unchanged.
    _ingest(client, graph_b, ["alpha"])
    nodes_b, _ = _all(client, graph_b)
    assert {node.uuid_ for node in nodes_b}.isdisjoint({node.uuid_ for node in nodes_a})
    assert _all(client, graph_a)[0] == nodes_a

    # A batch for graph A cannot take items for graph B.
    batch = client.batch.create(metadata={"graph_id": graph_a})
    with pytest.raises(ValueError):
        client.batch.add(
            batch_id=batch.batch_id,
            items=[BatchAddItem(type="graph_episode", graph_id=graph_b, data="alpha")],
        )

    # Deleting graph A leaves graph B intact.
    client.graph.delete(graph_a)
    assert len(_all(client, graph_b)[0]) == len(nodes_b)


def test_extraction_failure_surfaces_as_an_error(tmp_path):
    client = _client(tmp_path, StubLLM(fail=True))
    graph_id = _new_graph(client)

    batch = _ingest(client, graph_id, ["alpha", "beta"])

    assert batch.status == "failed"
    assert "provider unavailable" in batch.error
    items = client.batch.list_items(batch_id=batch.batch_id).items
    assert {item.status for item in items} == {"failed"}
    assert _all(client, graph_id) == ([], [])

    with pytest.raises(GraphExtractionError, match="provider unavailable"):
        client.graph.add(graph_id=graph_id, type="text", data="alpha")


def test_invalid_extraction_shape_is_an_error(tmp_path):
    client = _client(tmp_path, StubLLM(payload={"entities": "Alice", "relations": []}))
    graph_id = _new_graph(client)

    with pytest.raises(GraphExtractionError, match="invalid shape"):
        client.graph.add(graph_id=graph_id, type="text", data="alpha")


def test_batch_interrupted_by_a_restart_is_reported_as_failed(tmp_path):
    client = _client(tmp_path)
    graph_id = _new_graph(client)
    batch = client.batch.create(metadata={"graph_id": graph_id})
    connection = sqlite3.connect(client.store.db_path)
    with connection:
        connection.execute(
            "UPDATE batches SET status = 'processing' WHERE batch_id = ?", (batch.batch_id,)
        )
    connection.close()

    restarted = _client(tmp_path)

    summary = restarted.batch.get(batch_id=batch.batch_id)
    assert summary.status == "failed"
    assert "interrupted" in summary.error


def _builder(tmp_path, monkeypatch, llm):
    monkeypatch.setattr(Config, "GRAPH_DB_PATH", str(tmp_path / "builder.sqlite3"))
    zep.clear_zep_client_cache()
    builder = GraphBuilderService()
    builder.client.store._llm = llm
    real_sleep = time.sleep
    monkeypatch.setattr(graph_builder_module.time, "sleep", lambda _s: real_sleep(0.05))
    return builder


def test_graph_builder_end_to_end_reports_progress(tmp_path, monkeypatch):
    builder = _builder(tmp_path, monkeypatch, StubLLM())
    progress = []

    graph_id = builder.create_graph("Test")
    builder.set_ontology(graph_id, ONTOLOGY)
    submission = builder.add_text_batches(graph_id, ["alpha text", "beta text"], batch_size=1)
    episode_uuids = builder._wait_for_batch(submission, lambda msg, ratio: progress.append(ratio))
    data = builder.get_graph_data(graph_id)

    assert len(episode_uuids) == 2
    assert progress[-1] == 1.0
    assert data["node_count"] == 4
    assert data["edge_count"] == 3
    assert {edge["fact_type"] for edge in data["edges"]} == {"WORKS_AT", "RELATES_TO"}
    stored = builder.client.store.get_ontology(graph_id)
    assert [item["name"] for item in stored["entity_types"]] == ["Person", "Company"]

    builder.delete_graph(graph_id)
    with pytest.raises(NotFoundError):
        builder.client.graph.get(graph_id)
    zep.clear_zep_client_cache()


def test_graph_builder_raises_when_extraction_fails(tmp_path, monkeypatch):
    builder = _builder(tmp_path, monkeypatch, StubLLM(fail=True))

    graph_id = builder.create_graph("Test")
    builder.set_ontology(graph_id, ONTOLOGY)
    submission = builder.add_text_batches(graph_id, ["alpha text"])

    with pytest.raises(RuntimeError, match="ended as failed.*provider unavailable"):
        builder._wait_for_batch(submission)
    zep.clear_zep_client_cache()
