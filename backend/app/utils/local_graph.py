"""Local knowledge graph store (SQLite + FTS5) replacing Zep Cloud.

The store keeps graphs, nodes, edges, episodes and ingestion batches in one
SQLite file. Entity/relation extraction is done by the configured LLM
(``LLMClient``); nothing else leaves the machine.

``LocalGraphClient`` mirrors the small part of the Zep client surface that
MiroFish uses (``graph.*``, ``graph.node.*``, ``graph.edge.*``,
``graph.episode.get``, ``batch.*``) so the services change as little as
possible. Every read and write is scoped by ``graph_id``.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .logger import get_logger

logger = get_logger("mirofish.local_graph")

# Chunks per extraction call and the character budget for one call.
EXTRACTION_MAX_ITEMS_PER_CALL = 8
EXTRACTION_MAX_CHARS_PER_CALL = 6000
MAX_SUMMARY_CHARS = 2000
DEFAULT_EDGE_NAME = "RELATES_TO"


class NotFoundError(Exception):
    """A graph, node, edge, episode or batch does not exist in this graph."""


class GraphExtractionError(RuntimeError):
    """The LLM extraction step failed for an episode or batch."""


def sqlite_fts5_error() -> Optional[str]:
    """Return an error message if SQLite lacks FTS5 with the trigram tokenizer."""

    try:
        connection = sqlite3.connect(":memory:")
        try:
            connection.execute(
                "CREATE VIRTUAL TABLE fts_probe USING fts5(text, tokenize='trigram')"
            )
        finally:
            connection.close()
    except sqlite3.Error as error:
        return (
            "SQLite FTS5 with the trigram tokenizer is required for the local "
            f"graph store (SQLite {sqlite3.sqlite_version}, needs >= 3.34 built "
            f"with FTS5): {error}"
        )
    return None


# ---------------------------------------------------------------------------
# Returned objects (attribute names follow the Zep SDK objects the services
# already read).
# ---------------------------------------------------------------------------


@dataclass
class GraphRecord:
    graph_id: str
    name: str
    description: str
    created_at: str


@dataclass
class GraphNode:
    uuid_: str
    name: str
    labels: List[str]
    summary: str
    attributes: Dict[str, Any]
    created_at: str

    @property
    def uuid(self) -> str:
        return self.uuid_


@dataclass
class GraphEdge:
    uuid_: str
    name: str
    fact: str
    source_node_uuid: str
    target_node_uuid: str
    attributes: Dict[str, Any]
    created_at: str
    valid_at: Optional[str] = None
    invalid_at: Optional[str] = None
    expired_at: Optional[str] = None
    episodes: List[str] = field(default_factory=list)

    @property
    def uuid(self) -> str:
        return self.uuid_

    @property
    def fact_type(self) -> str:
        return self.name


@dataclass
class Episode:
    uuid_: str
    graph_id: str
    content: str
    source_description: str
    created_at: str
    processed: bool
    error: Optional[str] = None

    @property
    def uuid(self) -> str:
        return self.uuid_


@dataclass
class SearchResults:
    edges: List[GraphEdge] = field(default_factory=list)
    nodes: List[GraphNode] = field(default_factory=list)


@dataclass
class RawPage:
    data: List[Any]
    headers: Dict[str, str]


@dataclass
class BatchAddItem:
    type: str
    graph_id: str
    data: str
    data_type: str = "text"
    source_description: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class BatchItem:
    sequence_index: int
    status: str
    episode_uuid: Optional[str]
    error: Optional[str] = None
    source_uuid: Optional[str] = None


@dataclass
class BatchProgress:
    total_items: int
    succeeded_items: int
    failed_items: int
    percent_complete: float


@dataclass
class Batch:
    batch_id: str
    status: str
    metadata: Dict[str, Any]
    progress: BatchProgress
    error: Optional[str] = None


@dataclass
class ItemPage:
    items: List[BatchItem]
    next_cursor: Optional[int]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS graphs (
    graph_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    ontology_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS nodes (
    uuid TEXT PRIMARY KEY,
    graph_id TEXT NOT NULL,
    name TEXT NOT NULL,
    name_key TEXT NOT NULL,
    labels_json TEXT NOT NULL,
    summary TEXT NOT NULL DEFAULT '',
    attributes_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    UNIQUE (graph_id, name_key)
);
CREATE TABLE IF NOT EXISTS edges (
    uuid TEXT PRIMARY KEY,
    graph_id TEXT NOT NULL,
    name TEXT NOT NULL,
    fact TEXT NOT NULL,
    fact_key TEXT NOT NULL,
    source_node_uuid TEXT NOT NULL,
    target_node_uuid TEXT NOT NULL,
    attributes_json TEXT NOT NULL DEFAULT '{}',
    episodes_json TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL,
    valid_at TEXT,
    UNIQUE (graph_id, source_node_uuid, target_node_uuid, name, fact_key)
);
CREATE TABLE IF NOT EXISTS episodes (
    uuid TEXT PRIMARY KEY,
    graph_id TEXT NOT NULL,
    content TEXT NOT NULL,
    source_description TEXT NOT NULL DEFAULT '',
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    processed INTEGER NOT NULL DEFAULT 0,
    error TEXT
);
CREATE TABLE IF NOT EXISTS batches (
    batch_id TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    error TEXT
);
CREATE TABLE IF NOT EXISTS batch_items (
    batch_id TEXT NOT NULL,
    sequence_index INTEGER NOT NULL,
    graph_id TEXT NOT NULL,
    episode_uuid TEXT NOT NULL,
    status TEXT NOT NULL,
    error TEXT,
    PRIMARY KEY (batch_id, sequence_index)
);
CREATE INDEX IF NOT EXISTS idx_nodes_graph ON nodes (graph_id, uuid);
CREATE INDEX IF NOT EXISTS idx_edges_graph ON edges (graph_id, uuid);
CREATE INDEX IF NOT EXISTS idx_episodes_graph ON episodes (graph_id);
CREATE INDEX IF NOT EXISTS idx_batch_items_graph ON batch_items (graph_id);
CREATE VIRTUAL TABLE IF NOT EXISTS nodes_fts USING fts5(
    name, summary, uuid UNINDEXED, graph_id UNINDEXED, tokenize='trigram'
);
CREATE VIRTUAL TABLE IF NOT EXISTS edges_fts USING fts5(
    fact, name, uuid UNINDEXED, graph_id UNINDEXED, tokenize='trigram'
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _name_key(name: str) -> str:
    return re.sub(r"\s+", " ", name).strip().casefold()


def _edge_name(value: Any) -> str:
    text = re.sub(r"[^0-9A-Za-z]+", "_", str(value or "")).strip("_").upper()
    return text or DEFAULT_EDGE_NAME


def _merge_summary(existing: str, new: str) -> str:
    new = (new or "").strip()
    existing = (existing or "").strip()
    if not new or new in existing:
        return existing
    if not existing:
        return new[:MAX_SUMMARY_CHARS]
    return f"{existing} {new}"[:MAX_SUMMARY_CHARS]


def _query_terms(query: str) -> Tuple[List[str], List[str]]:
    """Split a query into FTS5 trigram phrases (>= 3 chars) and short terms."""

    long_terms: List[str] = []
    short_terms: List[str] = []
    for word in re.findall(r"\w+", query):
        word = word.strip()
        if len(word) >= 3:
            if word.casefold() not in {t.casefold() for t in long_terms}:
                long_terms.append(word)
        elif word and word not in short_terms:
            short_terms.append(word)
    return long_terms, short_terms


def _fts_match_expression(terms: Iterable[str]) -> str:
    return " OR ".join('"' + term.replace('"', '""') + '"' for term in terms)


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


class LocalGraphStore:
    """SQLite persistence and extraction for local graphs."""

    def __init__(self, db_path: str, llm: Any = None):
        self.db_path = db_path
        # An injected client serves every stage; otherwise one LLMClient per
        # stage is created on first use.
        self._llm = llm
        self._stage_llms: Dict[str, Any] = {}
        self._llm_guard = threading.Lock()
        self._write_lock = threading.RLock()
        self._active_batches: set[str] = set()
        self._active_guard = threading.Lock()
        error = sqlite_fts5_error()
        if error:
            raise RuntimeError(error)
        directory = os.path.dirname(os.path.abspath(db_path))
        os.makedirs(directory, exist_ok=True)
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.executescript(_SCHEMA)

    # -- infrastructure ---------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=30)
        connection.row_factory = sqlite3.Row
        return connection

    def _read(self, sql: str, params: Iterable[Any] = ()) -> List[sqlite3.Row]:
        connection = self._connect()
        try:
            return connection.execute(sql, tuple(params)).fetchall()
        finally:
            connection.close()

    def llm_for(self, stage: str) -> Any:
        """Return the LLM client for a pipeline stage.

        Uploaded document chunks (batch ingestion) use the "graph" stage;
        simulation activity written with ``graph.add`` uses "simulation".
        """
        if self._llm is not None:
            return self._llm
        with self._llm_guard:
            client = self._stage_llms.get(stage)
            if client is None:
                from .llm_client import LLMClient

                client = LLMClient(stage=stage)
                self._stage_llms[stage] = client
            return client

    # -- row mapping ------------------------------------------------------

    @staticmethod
    def _node(row: sqlite3.Row) -> GraphNode:
        return GraphNode(
            uuid_=row["uuid"],
            name=row["name"],
            labels=json.loads(row["labels_json"]),
            summary=row["summary"],
            attributes=json.loads(row["attributes_json"]),
            created_at=row["created_at"],
        )

    @staticmethod
    def _edge(row: sqlite3.Row) -> GraphEdge:
        return GraphEdge(
            uuid_=row["uuid"],
            name=row["name"],
            fact=row["fact"],
            source_node_uuid=row["source_node_uuid"],
            target_node_uuid=row["target_node_uuid"],
            attributes=json.loads(row["attributes_json"]),
            created_at=row["created_at"],
            valid_at=row["valid_at"],
            episodes=json.loads(row["episodes_json"]),
        )

    # -- graphs -----------------------------------------------------------

    def create_graph(self, graph_id: str, name: str, description: str = "") -> GraphRecord:
        created_at = _now()
        with self._write_lock, self._connect() as connection:
            try:
                connection.execute(
                    "INSERT INTO graphs (graph_id, name, description, created_at) "
                    "VALUES (?, ?, ?, ?)",
                    (graph_id, name, description or "", created_at),
                )
            except sqlite3.IntegrityError as error:
                raise ValueError(f"Graph already exists: {graph_id}") from error
        return GraphRecord(graph_id, name, description or "", created_at)

    def get_graph(self, graph_id: str) -> GraphRecord:
        rows = self._read("SELECT * FROM graphs WHERE graph_id = ?", (graph_id,))
        if not rows:
            raise NotFoundError(f"Graph not found: {graph_id}")
        row = rows[0]
        return GraphRecord(row["graph_id"], row["name"], row["description"], row["created_at"])

    def get_ontology(self, graph_id: str) -> Dict[str, Any]:
        rows = self._read("SELECT ontology_json FROM graphs WHERE graph_id = ?", (graph_id,))
        if not rows:
            raise NotFoundError(f"Graph not found: {graph_id}")
        return json.loads(rows[0]["ontology_json"])

    def set_ontology(self, graph_id: str, ontology: Dict[str, Any]) -> None:
        with self._write_lock, self._connect() as connection:
            cursor = connection.execute(
                "UPDATE graphs SET ontology_json = ? WHERE graph_id = ?",
                (json.dumps(ontology, ensure_ascii=False), graph_id),
            )
            if cursor.rowcount == 0:
                raise NotFoundError(f"Graph not found: {graph_id}")

    def delete_graph(self, graph_id: str) -> None:
        with self._write_lock, self._connect() as connection:
            cursor = connection.execute("DELETE FROM graphs WHERE graph_id = ?", (graph_id,))
            if cursor.rowcount == 0:
                raise NotFoundError(f"Graph not found: {graph_id}")
            for table in ("nodes", "edges", "episodes", "batch_items", "nodes_fts", "edges_fts"):
                connection.execute(f"DELETE FROM {table} WHERE graph_id = ?", (graph_id,))
            connection.execute(
                "DELETE FROM batches WHERE json_extract(metadata_json, '$.graph_id') = ?",
                (graph_id,),
            )

    def _require_graph(self, graph_id: str) -> None:
        self.get_graph(graph_id)

    # -- nodes / edges ----------------------------------------------------

    def get_node(self, graph_id: str, node_uuid: str) -> GraphNode:
        rows = self._read(
            "SELECT * FROM nodes WHERE graph_id = ? AND uuid = ?", (graph_id, node_uuid)
        )
        if not rows:
            raise NotFoundError(f"Node not found: {node_uuid}")
        return self._node(rows[0])

    def get_node_edges(self, graph_id: str, node_uuid: str) -> List[GraphEdge]:
        self.get_node(graph_id, node_uuid)
        rows = self._read(
            "SELECT * FROM edges WHERE graph_id = ? "
            "AND (source_node_uuid = ? OR target_node_uuid = ?) ORDER BY uuid",
            (graph_id, node_uuid, node_uuid),
        )
        return [self._edge(row) for row in rows]

    def list_page(
        self, table: str, graph_id: str, limit: int, cursor: Optional[str]
    ) -> RawPage:
        if table not in ("nodes", "edges"):
            raise ValueError(table)
        self._require_graph(graph_id)
        params: List[Any] = [graph_id]
        sql = f"SELECT * FROM {table} WHERE graph_id = ?"
        if cursor:
            sql += " AND uuid > ?"
            params.append(cursor)
        sql += " ORDER BY uuid LIMIT ?"
        params.append(limit + 1)
        rows = self._read(sql, params)
        has_more = len(rows) > limit
        rows = rows[:limit]
        mapper = self._node if table == "nodes" else self._edge
        data = [mapper(row) for row in rows]
        headers = {"zep-next-cursor": rows[-1]["uuid"]} if has_more and rows else {}
        return RawPage(data=data, headers=headers)

    def get_episode(self, graph_id: str, episode_uuid: str) -> Episode:
        rows = self._read(
            "SELECT * FROM episodes WHERE graph_id = ? AND uuid = ?",
            (graph_id, episode_uuid),
        )
        if not rows:
            raise NotFoundError(f"Episode not found: {episode_uuid}")
        row = rows[0]
        return Episode(
            uuid_=row["uuid"],
            graph_id=row["graph_id"],
            content=row["content"],
            source_description=row["source_description"],
            created_at=row["created_at"],
            processed=bool(row["processed"]),
            error=row["error"],
        )

    # -- search -----------------------------------------------------------

    def search(self, graph_id: str, query: str, limit: int, scope: str) -> SearchResults:
        self._require_graph(graph_id)
        if scope not in ("edges", "nodes"):
            raise ValueError("scope must be 'edges' or 'nodes'")
        long_terms, short_terms = _query_terms(query)
        table = "edges" if scope == "edges" else "nodes"
        fts = f"{table}_fts"
        uuids: List[str] = []
        if long_terms:
            rows = self._read(
                f"SELECT uuid FROM {fts} WHERE {fts} MATCH ? AND graph_id = ? "
                f"ORDER BY bm25({fts}) LIMIT ?",
                (_fts_match_expression(long_terms), graph_id, limit),
            )
            uuids.extend(row["uuid"] for row in rows)
        if short_terms and len(uuids) < limit:
            # Trigram FTS cannot match terms shorter than 3 characters
            # (e.g. two-character CJK names); fall back to substring matching.
            columns = ("fact", "name") if scope == "edges" else ("name", "summary")
            conditions = " OR ".join(
                f"{column} LIKE ? ESCAPE '\\'" for column in columns for _ in short_terms
            )
            patterns = [
                "%" + term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"
                for _ in columns
                for term in short_terms
            ]
            rows = self._read(
                f"SELECT uuid FROM {table} WHERE graph_id = ? AND ({conditions}) "
                "ORDER BY uuid LIMIT ?",
                [graph_id, *patterns, limit * 2],
            )
            for row in rows:
                if row["uuid"] not in uuids:
                    uuids.append(row["uuid"])
        uuids = uuids[:limit]
        if not uuids:
            return SearchResults()
        placeholders = ",".join("?" for _ in uuids)
        rows = self._read(
            f"SELECT * FROM {table} WHERE graph_id = ? AND uuid IN ({placeholders})",
            [graph_id, *uuids],
        )
        by_uuid = {row["uuid"]: row for row in rows}
        ordered = [by_uuid[item] for item in uuids if item in by_uuid]
        if scope == "edges":
            return SearchResults(edges=[self._edge(row) for row in ordered])
        return SearchResults(nodes=[self._node(row) for row in ordered])

    # -- extraction -------------------------------------------------------

    def _extraction_messages(
        self, ontology: Dict[str, Any], texts: List[str]
    ) -> List[Dict[str, str]]:
        system = (
            "You build a knowledge graph from text. Extract the entities and "
            "relationships the text states explicitly. Use only the entity "
            "types and relationship types of the ontology below; when an "
            "entity fits none of them, use the type \"Entity\". A "
            "relationship's source and target types should be one of its "
            "allowed source_targets (\"Entity\" means any type). Keep entity "
            "names exactly as written in the text. Write summaries and facts "
            "in the language of the text.\n\n"
            "Ontology:\n" + json.dumps(ontology, ensure_ascii=False, indent=2) + "\n\n"
            "Return one JSON object with this shape:\n"
            '{"entities": [{"name": str, "type": str, "summary": str, '
            '"attributes": {attribute_name: str}}], '
            '"relations": [{"source": entity name, "target": entity name, '
            '"type": str, "fact": one sentence stating the relationship}]}'
        )
        body = "\n\n".join(
            f"<text index=\"{index}\">\n{text}\n</text>" for index, text in enumerate(texts)
        )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": body},
        ]

    def extract(
        self, ontology: Dict[str, Any], texts: List[str], stage: str = "graph"
    ) -> Dict[str, Any]:
        """Call the ``stage`` LLM once for ``texts`` and validate the JSON shape."""

        try:
            result = self.llm_for(stage).chat_json(
                self._extraction_messages(ontology, texts),
                temperature=0.0,
                max_tokens=16000,
                max_attempts=2,
            )
        except Exception as error:
            raise GraphExtractionError(
                f"LLM extraction failed: {type(error).__name__}: {error}"
            ) from error
        entities = result.get("entities", [])
        relations = result.get("relations", [])
        if not isinstance(entities, list) or not isinstance(relations, list):
            raise GraphExtractionError(
                "LLM extraction returned an invalid shape "
                "(expected lists 'entities' and 'relations')"
            )
        return {"entities": entities, "relations": relations}

    @staticmethod
    def _ontology_index(ontology: Dict[str, Any]):
        entity_types = {
            str(item.get("name")): item
            for item in ontology.get("entity_types", []) or []
            if isinstance(item, dict) and item.get("name")
        }
        edge_types = {
            str(item.get("name")): item
            for item in ontology.get("edge_types", []) or []
            if isinstance(item, dict) and item.get("name")
        }
        return entity_types, edge_types

    def _write_extraction(
        self,
        connection: sqlite3.Connection,
        graph_id: str,
        ontology: Dict[str, Any],
        extraction: Dict[str, Any],
        episode_uuids: List[str],
        valid_at: str,
    ) -> None:
        entity_types, edge_types = self._ontology_index(ontology)
        now = _now()
        node_types: Dict[str, str] = {}
        node_uuids: Dict[str, str] = {}

        def upsert_node(name: str, entity_type: str, summary: str, attributes: Dict[str, Any]) -> str:
            key = _name_key(name)
            row = connection.execute(
                "SELECT * FROM nodes WHERE graph_id = ? AND name_key = ?", (graph_id, key)
            ).fetchone()
            labels = ["Entity"] + ([entity_type] if entity_type in entity_types else [])
            if row is None:
                node_uuid = str(uuid.uuid4())
                connection.execute(
                    "INSERT INTO nodes (uuid, graph_id, name, name_key, labels_json, "
                    "summary, attributes_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        node_uuid, graph_id, name, key,
                        json.dumps(labels, ensure_ascii=False),
                        (summary or "")[:MAX_SUMMARY_CHARS],
                        json.dumps(attributes, ensure_ascii=False), now,
                    ),
                )
                merged_summary = (summary or "")[:MAX_SUMMARY_CHARS]
            else:
                node_uuid = row["uuid"]
                existing_labels = json.loads(row["labels_json"])
                if len(existing_labels) > 1:
                    labels = existing_labels  # first typed label wins
                merged_attributes = json.loads(row["attributes_json"])
                merged_attributes.update(attributes)
                merged_summary = _merge_summary(row["summary"], summary)
                connection.execute(
                    "UPDATE nodes SET labels_json = ?, summary = ?, attributes_json = ? "
                    "WHERE uuid = ? AND graph_id = ?",
                    (
                        json.dumps(labels, ensure_ascii=False), merged_summary,
                        json.dumps(merged_attributes, ensure_ascii=False),
                        node_uuid, graph_id,
                    ),
                )
                connection.execute(
                    "DELETE FROM nodes_fts WHERE uuid = ? AND graph_id = ?", (node_uuid, graph_id)
                )
            connection.execute(
                "INSERT INTO nodes_fts (name, summary, uuid, graph_id) VALUES (?, ?, ?, ?)",
                (name if row is None else row["name"], merged_summary, node_uuid, graph_id),
            )
            node_types[key] = labels[1] if len(labels) > 1 else "Entity"
            node_uuids[key] = node_uuid
            return node_uuid

        for entity in extraction["entities"]:
            if not isinstance(entity, dict):
                continue
            name = str(entity.get("name") or "").strip()
            if not name:
                continue
            attributes = entity.get("attributes")
            attributes = {
                str(k): v for k, v in attributes.items()
                if v not in (None, "", [], {})
            } if isinstance(attributes, dict) else {}
            upsert_node(
                name,
                str(entity.get("type") or "Entity").strip(),
                str(entity.get("summary") or ""),
                attributes,
            )

        for relation in extraction["relations"]:
            if not isinstance(relation, dict):
                continue
            source = str(relation.get("source") or "").strip()
            target = str(relation.get("target") or "").strip()
            fact = str(relation.get("fact") or "").strip()
            if not source or not target or not fact:
                continue
            for endpoint in (source, target):
                if _name_key(endpoint) not in node_uuids:
                    existing = connection.execute(
                        "SELECT uuid, labels_json FROM nodes WHERE graph_id = ? AND name_key = ?",
                        (graph_id, _name_key(endpoint)),
                    ).fetchone()
                    if existing is None:
                        upsert_node(endpoint, "Entity", "", {})
                    else:
                        labels = json.loads(existing["labels_json"])
                        node_uuids[_name_key(endpoint)] = existing["uuid"]
                        node_types[_name_key(endpoint)] = labels[1] if len(labels) > 1 else "Entity"
            name = _edge_name(relation.get("type"))
            edge_def = edge_types.get(name)
            if edge_def is None:
                name = DEFAULT_EDGE_NAME
            else:
                allowed = edge_def.get("source_targets") or []
                pair = (node_types[_name_key(source)], node_types[_name_key(target)])
                if allowed and not any(
                    st.get("source") in ("Entity", pair[0])
                    and st.get("target") in ("Entity", pair[1])
                    for st in allowed
                ):
                    name = DEFAULT_EDGE_NAME
            source_uuid = node_uuids[_name_key(source)]
            target_uuid = node_uuids[_name_key(target)]
            fact_key = _name_key(fact)
            row = connection.execute(
                "SELECT uuid, episodes_json FROM edges WHERE graph_id = ? AND source_node_uuid = ? "
                "AND target_node_uuid = ? AND name = ? AND fact_key = ?",
                (graph_id, source_uuid, target_uuid, name, fact_key),
            ).fetchone()
            if row is not None:
                episodes = json.loads(row["episodes_json"])
                episodes.extend(e for e in episode_uuids if e not in episodes)
                connection.execute(
                    "UPDATE edges SET episodes_json = ? WHERE uuid = ? AND graph_id = ?",
                    (json.dumps(episodes), row["uuid"], graph_id),
                )
                continue
            edge_uuid = str(uuid.uuid4())
            connection.execute(
                "INSERT INTO edges (uuid, graph_id, name, fact, fact_key, source_node_uuid, "
                "target_node_uuid, attributes_json, episodes_json, created_at, valid_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, '{}', ?, ?, ?)",
                (
                    edge_uuid, graph_id, name, fact, fact_key, source_uuid, target_uuid,
                    json.dumps(episode_uuids), now, valid_at,
                ),
            )
            connection.execute(
                "INSERT INTO edges_fts (fact, name, uuid, graph_id) VALUES (?, ?, ?, ?)",
                (fact, name, edge_uuid, graph_id),
            )

    def _insert_episode(
        self,
        connection: sqlite3.Connection,
        graph_id: str,
        content: str,
        source_description: str,
        metadata: Optional[Dict[str, Any]],
        created_at: Optional[str],
    ) -> str:
        episode_uuid = str(uuid.uuid4())
        connection.execute(
            "INSERT INTO episodes (uuid, graph_id, content, source_description, "
            "metadata_json, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (
                episode_uuid, graph_id, content, source_description or "",
                json.dumps(metadata or {}, ensure_ascii=False), created_at or _now(),
            ),
        )
        return episode_uuid

    def add_episode(
        self,
        graph_id: str,
        data: str,
        *,
        created_at: Optional[str] = None,
        source_description: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Episode:
        """Store one text episode and extract it synchronously."""

        self._require_graph(graph_id)
        with self._write_lock, self._connect() as connection:
            episode_uuid = self._insert_episode(
                connection, graph_id, data, source_description, metadata, created_at
            )
        ontology = self.get_ontology(graph_id)
        try:
            # graph.add is only called for simulated agent activity (high
            # volume), so it uses the simulation model, not the graph model.
            extraction = self.extract(ontology, [data], stage="simulation")
            with self._write_lock, self._connect() as connection:
                self._write_extraction(
                    connection, graph_id, ontology, extraction, [episode_uuid],
                    created_at or _now(),
                )
                connection.execute(
                    "UPDATE episodes SET processed = 1 WHERE uuid = ? AND graph_id = ?",
                    (episode_uuid, graph_id),
                )
        except Exception as error:
            with self._write_lock, self._connect() as connection:
                connection.execute(
                    "UPDATE episodes SET error = ? WHERE uuid = ? AND graph_id = ?",
                    (str(error)[:1000], episode_uuid, graph_id),
                )
            if isinstance(error, GraphExtractionError):
                raise
            raise GraphExtractionError(str(error)) from error
        return self.get_episode(graph_id, episode_uuid)

    # -- batches ----------------------------------------------------------

    def create_batch(self, metadata: Dict[str, Any]) -> Batch:
        graph_id = metadata.get("graph_id")
        if not graph_id:
            raise ValueError("batch metadata must include graph_id")
        self._require_graph(graph_id)
        batch_id = str(uuid.uuid4())
        with self._write_lock, self._connect() as connection:
            connection.execute(
                "INSERT INTO batches (batch_id, status, metadata_json, created_at) "
                "VALUES (?, 'draft', ?, ?)",
                (batch_id, json.dumps(metadata, ensure_ascii=False), _now()),
            )
        return self.get_batch(batch_id)

    def _batch_row(self, batch_id: str) -> sqlite3.Row:
        rows = self._read("SELECT * FROM batches WHERE batch_id = ?", (batch_id,))
        if not rows:
            raise NotFoundError(f"Batch not found: {batch_id}")
        return rows[0]

    def add_batch_items(self, batch_id: str, items: List[BatchAddItem]) -> List[BatchItem]:
        batch = self._batch_row(batch_id)
        if batch["status"] != "draft":
            raise ValueError(f"Batch {batch_id} is not a draft")
        graph_id = json.loads(batch["metadata_json"])["graph_id"]
        details: List[BatchItem] = []
        with self._write_lock, self._connect() as connection:
            next_index = connection.execute(
                "SELECT COALESCE(MAX(sequence_index) + 1, 0) FROM batch_items WHERE batch_id = ?",
                (batch_id,),
            ).fetchone()[0]
            for offset, item in enumerate(items):
                if item.graph_id != graph_id:
                    raise ValueError("All batch items must target the batch graph")
                episode_uuid = self._insert_episode(
                    connection, graph_id, item.data, item.source_description or "",
                    item.metadata, None,
                )
                connection.execute(
                    "INSERT INTO batch_items (batch_id, sequence_index, graph_id, "
                    "episode_uuid, status) VALUES (?, ?, ?, ?, 'pending')",
                    (batch_id, next_index + offset, graph_id, episode_uuid),
                )
                details.append(BatchItem(next_index + offset, "pending", episode_uuid))
        return details

    def process_batch(self, batch_id: str) -> Batch:
        batch = self._batch_row(batch_id)
        if batch["status"] != "draft":
            raise ValueError(f"Batch {batch_id} is not a draft")
        with self._write_lock, self._connect() as connection:
            connection.execute(
                "UPDATE batches SET status = 'processing' WHERE batch_id = ?", (batch_id,)
            )
        with self._active_guard:
            self._active_batches.add(batch_id)
        thread = threading.Thread(
            target=self._run_batch, args=(batch_id,), daemon=True,
            name=f"graph-batch-{batch_id[:8]}",
        )
        thread.start()
        return self.get_batch(batch_id)

    def _groups(self, rows: List[sqlite3.Row]) -> List[List[sqlite3.Row]]:
        groups: List[List[sqlite3.Row]] = []
        current: List[sqlite3.Row] = []
        size = 0
        for row in rows:
            length = len(row["content"])
            if current and (
                len(current) >= EXTRACTION_MAX_ITEMS_PER_CALL
                or size + length > EXTRACTION_MAX_CHARS_PER_CALL
            ):
                groups.append(current)
                current, size = [], 0
            current.append(row)
            size += length
        if current:
            groups.append(current)
        return groups

    def _run_batch(self, batch_id: str) -> None:
        try:
            batch = self._batch_row(batch_id)
            graph_id = json.loads(batch["metadata_json"])["graph_id"]
            ontology = self.get_ontology(graph_id)
            rows = self._read(
                "SELECT bi.sequence_index, bi.episode_uuid, e.content, e.created_at "
                "FROM batch_items bi JOIN episodes e "
                "ON e.uuid = bi.episode_uuid AND e.graph_id = bi.graph_id "
                "WHERE bi.batch_id = ? AND bi.graph_id = ? AND bi.status = 'pending' "
                "ORDER BY bi.sequence_index",
                (batch_id, graph_id),
            )
            for group in self._groups(rows):
                indexes = [row["sequence_index"] for row in group]
                episode_uuids = [row["episode_uuid"] for row in group]
                try:
                    extraction = self.extract(
                        ontology, [row["content"] for row in group], stage="graph"
                    )
                    with self._write_lock, self._connect() as connection:
                        self._write_extraction(
                            connection, graph_id, ontology, extraction, episode_uuids,
                            group[0]["created_at"],
                        )
                        connection.executemany(
                            "UPDATE episodes SET processed = 1 WHERE uuid = ? AND graph_id = ?",
                            [(episode, graph_id) for episode in episode_uuids],
                        )
                        connection.executemany(
                            "UPDATE batch_items SET status = 'succeeded' "
                            "WHERE batch_id = ? AND sequence_index = ?",
                            [(batch_id, index) for index in indexes],
                        )
                except Exception as error:
                    message = str(error)[:1000]
                    logger.error(
                        "Graph batch %s extraction failed (%s)", batch_id, type(error).__name__
                    )
                    with self._write_lock, self._connect() as connection:
                        connection.executemany(
                            "UPDATE batch_items SET status = 'failed', error = ? "
                            "WHERE batch_id = ? AND sequence_index = ?",
                            [(message, batch_id, index) for index in indexes],
                        )
                        connection.executemany(
                            "UPDATE episodes SET error = ? WHERE uuid = ? AND graph_id = ?",
                            [(message, episode, graph_id) for episode in episode_uuids],
                        )
                        connection.execute(
                            "UPDATE batches SET status = 'failed', error = ? WHERE batch_id = ?",
                            (message, batch_id),
                        )
                    return
            with self._write_lock, self._connect() as connection:
                connection.execute(
                    "UPDATE batches SET status = 'succeeded' WHERE batch_id = ?", (batch_id,)
                )
        except Exception as error:  # store/graph errors outside extraction
            logger.error("Graph batch %s failed (%s)", batch_id, type(error).__name__)
            with self._write_lock, self._connect() as connection:
                connection.execute(
                    "UPDATE batches SET status = 'failed', error = ? WHERE batch_id = ?",
                    (str(error)[:1000], batch_id),
                )
        finally:
            with self._active_guard:
                self._active_batches.discard(batch_id)

    def get_batch(self, batch_id: str) -> Batch:
        row = self._batch_row(batch_id)
        status = row["status"]
        if status == "processing":
            with self._active_guard:
                active = batch_id in self._active_batches
            if not active:
                # The worker belonged to an earlier process. Report it as
                # failed instead of letting callers wait forever.
                message = "Batch processing was interrupted (process restarted)"
                with self._write_lock, self._connect() as connection:
                    connection.execute(
                        "UPDATE batches SET status = 'failed', error = ? "
                        "WHERE batch_id = ? AND status = 'processing'",
                        (message, batch_id),
                    )
                row = self._batch_row(batch_id)
                status = row["status"]
        counts = self._read(
            "SELECT status, COUNT(*) AS n FROM batch_items WHERE batch_id = ? GROUP BY status",
            (batch_id,),
        )
        by_status = {item["status"]: item["n"] for item in counts}
        total = sum(by_status.values())
        succeeded = by_status.get("succeeded", 0)
        return Batch(
            batch_id=batch_id,
            status=status,
            metadata=json.loads(row["metadata_json"]),
            progress=BatchProgress(
                total_items=total,
                succeeded_items=succeeded,
                failed_items=by_status.get("failed", 0),
                percent_complete=(100.0 * succeeded / total) if total else 0.0,
            ),
            error=row["error"],
        )

    def list_batch_items(self, batch_id: str, limit: int, cursor: Optional[int]) -> ItemPage:
        self._batch_row(batch_id)
        start = int(cursor or 0)
        rows = self._read(
            "SELECT * FROM batch_items WHERE batch_id = ? AND sequence_index >= ? "
            "ORDER BY sequence_index LIMIT ?",
            (batch_id, start, limit + 1),
        )
        has_more = len(rows) > limit
        rows = rows[:limit]
        items = [
            BatchItem(row["sequence_index"], row["status"], row["episode_uuid"], row["error"])
            for row in rows
        ]
        next_cursor = rows[-1]["sequence_index"] + 1 if has_more and rows else None
        return ItemPage(items=items, next_cursor=next_cursor)


# ---------------------------------------------------------------------------
# Zep-shaped client facade
# ---------------------------------------------------------------------------


class _RawListApi:
    def __init__(self, store: LocalGraphStore, table: str):
        self._store = store
        self._table = table

    def get_by_graph_id(self, graph_id: str, *, limit: int = 100, cursor: Optional[str] = None) -> RawPage:
        return self._store.list_page(self._table, graph_id, limit, cursor)


class _NodeApi:
    def __init__(self, store: LocalGraphStore):
        self._store = store
        self.with_raw_response = _RawListApi(store, "nodes")

    def get(self, *, uuid_: str, graph_id: str) -> GraphNode:
        return self._store.get_node(graph_id, uuid_)

    def get_edges(self, *, node_uuid: str, graph_id: str) -> List[GraphEdge]:
        return self._store.get_node_edges(graph_id, node_uuid)

    def get_by_graph_id(self, graph_id: str, *, limit: int = 100, cursor: Optional[str] = None) -> List[GraphNode]:
        return self._store.list_page("nodes", graph_id, limit, cursor).data


class _EdgeApi:
    def __init__(self, store: LocalGraphStore):
        self._store = store
        self.with_raw_response = _RawListApi(store, "edges")

    def get_by_graph_id(self, graph_id: str, *, limit: int = 100, cursor: Optional[str] = None) -> List[GraphEdge]:
        return self._store.list_page("edges", graph_id, limit, cursor).data


class _EpisodeApi:
    def __init__(self, store: LocalGraphStore):
        self._store = store

    def get(self, *, uuid_: str, graph_id: str) -> Episode:
        return self._store.get_episode(graph_id, uuid_)


class _GraphApi:
    def __init__(self, store: LocalGraphStore):
        self._store = store
        self.node = _NodeApi(store)
        self.edge = _EdgeApi(store)
        self.episode = _EpisodeApi(store)

    def create(self, *, graph_id: str, name: str, description: str = "") -> GraphRecord:
        return self._store.create_graph(graph_id, name, description)

    def get(self, graph_id: str) -> GraphRecord:
        return self._store.get_graph(graph_id)

    def delete(self, graph_id: str) -> None:
        self._store.delete_graph(graph_id)

    def set_ontology(self, *, graph_ids: List[str], ontology: Dict[str, Any]) -> None:
        for graph_id in graph_ids:
            self._store.set_ontology(graph_id, ontology)

    def add(
        self,
        *,
        graph_id: str,
        data: str,
        type: str = "text",
        created_at: Optional[str] = None,
        source_description: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Episode:
        if type != "text":
            raise ValueError("Only text episodes are supported")
        return self._store.add_episode(
            graph_id, data, created_at=created_at,
            source_description=source_description, metadata=metadata,
        )

    def search(
        self,
        *,
        graph_id: str,
        query: str,
        limit: int = 10,
        scope: str = "edges",
        reranker: Optional[str] = None,  # accepted for compatibility, ignored
    ) -> SearchResults:
        return self._store.search(graph_id, query, limit, scope)


class _BatchApi:
    def __init__(self, store: LocalGraphStore):
        self._store = store

    def create(self, *, metadata: Dict[str, Any]) -> Batch:
        return self._store.create_batch(metadata)

    def add(self, *, batch_id: str, items: List[BatchAddItem]) -> List[BatchItem]:
        return self._store.add_batch_items(batch_id, items)

    def process(self, *, batch_id: str) -> Batch:
        return self._store.process_batch(batch_id)

    def get(self, *, batch_id: str) -> Batch:
        return self._store.get_batch(batch_id)

    def list_items(self, *, batch_id: str, limit: int = 100, cursor: Optional[int] = None) -> ItemPage:
        return self._store.list_batch_items(batch_id, limit, cursor)


class LocalGraphClient:
    """Drop-in replacement for the parts of the Zep client MiroFish uses."""

    def __init__(self, db_path: str, llm: Any = None):
        self.store = LocalGraphStore(db_path, llm=llm)
        self.graph = _GraphApi(self.store)
        self.batch = _BatchApi(self.store)
