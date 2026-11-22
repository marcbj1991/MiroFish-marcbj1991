"""Per-stage model selection: the resolver and the stage of each call site.

The LLM SDK client is a stub that records the model of every request; the
graph store tests use a real SQLite file (no database mock).
"""

import json
from types import SimpleNamespace

import pytest

from app.config import (
    ANTHROPIC_STAGE_DEFAULT_MODELS,
    OPENAI_DEFAULT_MODEL,
    Config,
    resolve_stage_model,
)
from app.services import oasis_profile_generator as profile_module
from app.services import simulation_config_generator as config_module
from app.services import zep_graph_memory_updater as updater_module
from app.services.graph_builder import GraphBuilderService
from app.services.oasis_profile_generator import OasisProfileGenerator
from app.services.ontology_generator import OntologyGenerator
from app.services.report_agent import ReportAgent
from app.services.simulation_config_generator import SimulationConfigGenerator
from app.services.zep_graph_memory_updater import AgentActivity, ZepGraphMemoryUpdater
from app.utils import llm_client as llm_client_module
from app.utils import zep

STAGE_VARS = ("LLM_MODEL_GRAPH", "LLM_MODEL_SIMULATION", "LLM_MODEL_REPORT")
ALL_MODEL_VARS = STAGE_VARS + ("LLM_MODEL_NAME",)

STAGE_ENV = {
    "graph": "LLM_MODEL_GRAPH",
    "simulation": "LLM_MODEL_SIMULATION",
    "report": "LLM_MODEL_REPORT",
}

ONTOLOGY = {
    "entity_types": [{"name": "Person", "description": "A person", "attributes": []}],
    "edge_types": [],
}
EXTRACTION = {
    "entities": [{"name": "Alice", "type": "Person", "summary": "Alice.", "attributes": {}}],
    "relations": [],
}


@pytest.fixture(autouse=True)
def clean_model_env(monkeypatch):
    for name in ALL_MODEL_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(Config, "LLM_PROVIDER", "anthropic")
    monkeypatch.setattr(Config, "LLM_API_KEY", "test-key")


class RecordingSDK:
    """Stub SDK client: records ``model`` and returns a fixed JSON reply."""

    def __init__(self, payload):
        self.models = []
        self.payload = payload
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.models.append(kwargs["model"])
        return SimpleNamespace(
            model=kwargs["model"],
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=json.dumps(self.payload)),
                    finish_reason="stop",
                )
            ],
        )


def _record_sdk(monkeypatch, module, payload=None):
    sdk = RecordingSDK(payload if payload is not None else {"ok": True})
    monkeypatch.setattr(module, "create_llm_sdk_client", lambda *args, **kwargs: sdk)
    return sdk


def _distinct_stage_models(monkeypatch):
    monkeypatch.setenv("LLM_MODEL_GRAPH", "model-graph")
    monkeypatch.setenv("LLM_MODEL_SIMULATION", "model-simulation")
    monkeypatch.setenv("LLM_MODEL_REPORT", "model-report")
    monkeypatch.setenv("LLM_MODEL_NAME", "model-shared")


def _local_graph(tmp_path, monkeypatch):
    monkeypatch.setattr(Config, "GRAPH_DB_PATH", str(tmp_path / "graph.sqlite3"))
    zep.clear_zep_client_cache()
    client = zep.get_zep_client()
    graph_id = "graph-stage-test"
    client.graph.create(graph_id=graph_id, name="Stage test")
    client.graph.set_ontology(graph_ids=[graph_id], ontology=ONTOLOGY)
    return client, graph_id


# -- resolver -----------------------------------------------------------------


@pytest.mark.parametrize("stage", ["graph", "simulation", "report"])
@pytest.mark.parametrize("provider", ["anthropic", "openai"])
def test_stage_variable_wins(monkeypatch, stage, provider):
    monkeypatch.setenv("LLM_MODEL_NAME", "shared-model")
    monkeypatch.setenv(STAGE_ENV[stage], "stage-model")

    assert resolve_stage_model(stage, provider=provider) == "stage-model"
    assert resolve_stage_model(stage, provider=provider, default="config-model") == "stage-model"


@pytest.mark.parametrize("stage", ["graph", "simulation", "report"])
@pytest.mark.parametrize("provider", ["anthropic", "openai"])
def test_shared_model_name_applies_when_stage_variable_is_unset_or_empty(
    monkeypatch, stage, provider
):
    monkeypatch.setenv("LLM_MODEL_NAME", "shared-model")
    monkeypatch.setenv(STAGE_ENV[stage], "  ")

    assert resolve_stage_model(stage, provider=provider) == "shared-model"
    assert resolve_stage_model(stage, provider=provider, default="config-model") == "shared-model"


@pytest.mark.parametrize("stage", ["graph", "simulation", "report"])
def test_anthropic_built_in_stage_defaults(monkeypatch, stage):
    monkeypatch.setenv("LLM_MODEL_NAME", "")

    assert resolve_stage_model(stage, provider="anthropic") == ANTHROPIC_STAGE_DEFAULT_MODELS[stage]


def test_anthropic_default_ids_are_exact():
    assert ANTHROPIC_STAGE_DEFAULT_MODELS == {
        "graph": "claude-sonnet-5-5",
        "simulation": "claude-haiku-5-5",
        "report": "claude-opus-5-5",
    }


@pytest.mark.parametrize("stage", ["graph", "simulation", "report"])
def test_openai_keeps_the_upstream_fallback_for_every_stage(stage):
    assert resolve_stage_model(stage, provider="openai") == OPENAI_DEFAULT_MODEL == "gpt-4o-mini"


@pytest.mark.parametrize("provider", ["anthropic", "openai"])
def test_explicit_default_replaces_the_built_in_default_only(provider):
    assert resolve_stage_model("simulation", provider=provider, default="config-model") == "config-model"
    assert resolve_stage_model("simulation", provider=provider, default="") == (
        "claude-haiku-5-5" if provider == "anthropic" else "gpt-4o-mini"
    )


def test_provider_defaults_to_config(monkeypatch):
    assert resolve_stage_model("graph") == "claude-sonnet-5-5"
    monkeypatch.setattr(Config, "LLM_PROVIDER", "openai")
    assert resolve_stage_model("graph") == "gpt-4o-mini"


def test_unknown_stage_is_rejected():
    with pytest.raises(ValueError, match="Unknown LLM stage"):
        resolve_stage_model("ontology")


def test_llm_client_explicit_model_still_wins(monkeypatch):
    _distinct_stage_models(monkeypatch)
    _record_sdk(monkeypatch, llm_client_module)

    assert llm_client_module.LLMClient(model="explicit", stage="graph").model == "explicit"
    assert llm_client_module.LLMClient(stage="report").model == "model-report"


# -- call sites -----------------------------------------------------------------


def test_ontology_generator_uses_the_graph_stage(monkeypatch):
    _distinct_stage_models(monkeypatch)
    sdk = _record_sdk(monkeypatch, llm_client_module)

    generator = OntologyGenerator()
    generator.llm_client.chat_json([{"role": "user", "content": "x"}])

    assert generator.llm_client.model == "model-graph"
    assert sdk.models == ["model-graph"]


def test_document_batch_ingestion_uses_the_graph_stage(tmp_path, monkeypatch):
    _distinct_stage_models(monkeypatch)
    sdk = _record_sdk(monkeypatch, llm_client_module, EXTRACTION)
    monkeypatch.setattr(Config, "GRAPH_DB_PATH", str(tmp_path / "builder.sqlite3"))
    zep.clear_zep_client_cache()

    builder = GraphBuilderService()
    graph_id = builder.create_graph("Docs")
    builder.set_ontology(graph_id, ONTOLOGY)
    submission = builder.add_text_batches(graph_id, ["Alice text"], batch_size=1)
    builder._wait_for_batch(submission, lambda *_args: None)

    assert sdk.models == ["model-graph"]
    nodes = builder.client.graph.node.get_by_graph_id(graph_id)
    assert [node.name for node in nodes] == ["Alice"]


def test_memory_updater_graph_add_uses_the_simulation_stage(tmp_path, monkeypatch):
    _distinct_stage_models(monkeypatch)
    sdk = _record_sdk(monkeypatch, llm_client_module, EXTRACTION)
    client, graph_id = _local_graph(tmp_path, monkeypatch)
    monkeypatch.setattr(updater_module, "get_zep_client", lambda: client)

    updater = ZepGraphMemoryUpdater(graph_id, simulation_id="sim-stage")
    updater.SEND_INTERVAL = 0
    updater.start()
    updater.add_activity(
        AgentActivity(
            platform="twitter",
            agent_id=1,
            agent_name="Alice",
            action_type="CREATE_POST",
            action_args={"content": "hello"},
            round_num=1,
            timestamp="2026-07-22T12:00:00+08:00",
        )
    )
    updater.stop()

    assert sdk.models == ["model-simulation"]
    nodes = client.graph.node.get_by_graph_id(graph_id)
    assert [node.name for node in nodes] == ["Alice"]


def test_local_store_keeps_one_client_per_stage(tmp_path, monkeypatch):
    _distinct_stage_models(monkeypatch)
    _record_sdk(monkeypatch, llm_client_module, EXTRACTION)
    client, _graph_id = _local_graph(tmp_path, monkeypatch)

    assert client.store.llm_for("graph").model == "model-graph"
    assert client.store.llm_for("simulation").model == "model-simulation"
    assert client.store.llm_for("graph") is client.store.llm_for("graph")


def test_profile_generator_uses_the_simulation_stage(tmp_path, monkeypatch):
    _distinct_stage_models(monkeypatch)
    sdk = _record_sdk(monkeypatch, profile_module, {"bio": "b", "persona": "p"})
    monkeypatch.setattr(Config, "GRAPH_DB_PATH", str(tmp_path / "graph.sqlite3"))
    zep.clear_zep_client_cache()

    generator = OasisProfileGenerator()
    generator._generate_profile_with_llm("Alice", "student", "Alice studies.", {}, "")

    assert generator.model_name == "model-simulation"
    assert sdk.models and set(sdk.models) == {"model-simulation"}


def test_config_generator_uses_the_simulation_stage(monkeypatch):
    _distinct_stage_models(monkeypatch)
    sdk = _record_sdk(monkeypatch, config_module, {"ok": True})

    generator = SimulationConfigGenerator()
    assert generator._call_llm_with_retry("prompt", "system") == {"ok": True}

    assert generator.model_name == "model-simulation"
    assert sdk.models == ["model-simulation"]


def test_simulation_stage_default_is_haiku_on_anthropic(monkeypatch):
    sdk = _record_sdk(monkeypatch, config_module, {"ok": True})

    generator = SimulationConfigGenerator()
    generator._call_llm_with_retry("prompt", "system")

    assert sdk.models == ["claude-haiku-5-5"]


def test_report_agent_and_its_tools_use_the_report_stage(tmp_path, monkeypatch):
    _distinct_stage_models(monkeypatch)
    sdk = _record_sdk(monkeypatch, llm_client_module)
    monkeypatch.setattr(Config, "GRAPH_DB_PATH", str(tmp_path / "graph.sqlite3"))
    zep.clear_zep_client_cache()

    agent = ReportAgent("graph-1", "sim-1", "requirement")
    agent.llm.chat([{"role": "user", "content": "x"}])
    agent.zep_tools.llm.chat([{"role": "user", "content": "y"}])

    assert agent.llm.model == "model-report"
    assert agent.zep_tools.llm.model == "model-report"
    assert sdk.models == ["model-report", "model-report"]


def test_shared_model_name_alone_means_one_model_everywhere(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_MODEL_NAME", "one-model")
    _record_sdk(monkeypatch, llm_client_module)
    _record_sdk(monkeypatch, config_module)
    monkeypatch.setattr(Config, "GRAPH_DB_PATH", str(tmp_path / "graph.sqlite3"))
    zep.clear_zep_client_cache()

    assert OntologyGenerator().llm_client.model == "one-model"
    assert SimulationConfigGenerator().model_name == "one-model"
    assert ReportAgent("g", "s", "r").llm.model == "one-model"
    assert zep.get_zep_client().store.llm_for("simulation").model == "one-model"
