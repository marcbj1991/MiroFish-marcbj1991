from types import SimpleNamespace

import pytest

from app.services.zep_entity_reader import ZepEntityReader


def test_get_node_edges_reads_within_the_given_graph():
    calls = []

    class NodeApi:
        def get_edges(self, *, node_uuid, graph_id):
            calls.append((node_uuid, graph_id))
            return [SimpleNamespace(
                uuid_="edge-1",
                name="KNOWS",
                fact="Alice knows Bob",
                source_node_uuid="node-1",
                target_node_uuid="node-2",
                attributes={"since": "2024"},
            )]

    class GraphApi:
        node = NodeApi()

    class Client:
        graph = GraphApi()

    reader = object.__new__(ZepEntityReader)
    reader.client = Client()

    assert reader.get_node_edges("node-1", graph_id="graph-1") == [{
        "uuid": "edge-1",
        "name": "KNOWS",
        "fact": "Alice knows Bob",
        "source_node_uuid": "node-1",
        "target_node_uuid": "node-2",
        "attributes": {"since": "2024"},
    }]
    assert calls == [("node-1", "graph-1")]


def test_get_node_edges_requires_a_graph_id():
    reader = object.__new__(ZepEntityReader)
    reader.client = None

    with pytest.raises(ValueError, match="graph_id"):
        reader.get_node_edges("node-1")
