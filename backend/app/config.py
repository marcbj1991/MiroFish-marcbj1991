"""
Configuration management.
Loads configuration from the .env file in the project root.
"""

import os
from dotenv import load_dotenv

# Load the .env file from the project root
# Path: MiroFish/.env (relative to backend/app/config.py)
project_root_env = os.path.join(os.path.dirname(__file__), '../../.env')

if os.path.exists(project_root_env):
    load_dotenv(project_root_env, override=True)
else:
    # No .env in the project root: fall back to environment variables (production)
    load_dotenv(override=True)


class Config:
    """Flask configuration class"""
    
    # Flask settings
    SECRET_KEY = os.environ.get('SECRET_KEY', 'mirofish-secret-key')
    DEBUG = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    
    # JSON settings - disable ASCII escaping so non-ASCII text is returned as is
    JSON_AS_ASCII = False
    
    # LLM provider: "anthropic" (default, official Anthropic SDK) or "openai"
    # (any OpenAI-compatible server, e.g. a local Ollama endpoint).
    LLM_PROVIDER = os.environ.get('LLM_PROVIDER', 'anthropic').strip().lower()
    LLM_API_KEY = os.environ.get('LLM_API_KEY')
    # Only used by the "openai" provider.
    LLM_BASE_URL = os.environ.get('LLM_BASE_URL', 'https://api.openai.com/v1')
    LLM_MODEL_NAME = os.environ.get(
        'LLM_MODEL_NAME',
        'claude-opus-5-5' if LLM_PROVIDER == 'anthropic' else 'gpt-4o-mini',
    )
    # Per-stage models (optional): LLM_MODEL_GRAPH, LLM_MODEL_SIMULATION,
    # LLM_MODEL_REPORT. Resolve them with resolve_stage_model(), never here.
    
    # Local knowledge graph store (SQLite file, replaces Zep Cloud)
    GRAPH_DB_PATH = os.path.join(os.path.dirname(__file__), '../uploads/graph/graph.sqlite3')
    
    # File upload settings
    MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB
    UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), '../uploads')
    ALLOWED_EXTENSIONS = {'pdf', 'md', 'txt', 'markdown'}
    
    # Text processing settings
    DEFAULT_CHUNK_SIZE = 500  # Default chunk size
    DEFAULT_CHUNK_OVERLAP = 50  # Default chunk overlap
    
    # OASIS simulation settings
    OASIS_DEFAULT_MAX_ROUNDS = int(os.environ.get('OASIS_DEFAULT_MAX_ROUNDS', '10'))
    OASIS_SIMULATION_DATA_DIR = os.path.join(os.path.dirname(__file__), '../uploads/simulations')
    
    # Actions available on each OASIS platform
    OASIS_TWITTER_ACTIONS = [
        'CREATE_POST', 'LIKE_POST', 'REPOST', 'FOLLOW', 'DO_NOTHING', 'QUOTE_POST'
    ]
    OASIS_REDDIT_ACTIONS = [
        'LIKE_POST', 'DISLIKE_POST', 'CREATE_POST', 'CREATE_COMMENT',
        'LIKE_COMMENT', 'DISLIKE_COMMENT', 'SEARCH_POSTS', 'SEARCH_USER',
        'TREND', 'REFRESH', 'DO_NOTHING', 'FOLLOW', 'MUTE'
    ]
    
    # Report Agent settings
    REPORT_AGENT_MAX_TOOL_CALLS = int(os.environ.get('REPORT_AGENT_MAX_TOOL_CALLS', '5'))
    REPORT_AGENT_MAX_REFLECTION_ROUNDS = int(os.environ.get('REPORT_AGENT_MAX_REFLECTION_ROUNDS', '2'))
    REPORT_AGENT_TEMPERATURE = float(os.environ.get('REPORT_AGENT_TEMPERATURE', '0.5'))
    
    @classmethod
    def validate(cls) -> list[str]:
        """Validate the required configuration"""
        errors: list[str] = []
        if not cls.LLM_API_KEY:
            errors.append("LLM_API_KEY is not configured")
        if cls.LLM_PROVIDER not in ("anthropic", "openai"):
            errors.append("LLM_PROVIDER must be 'anthropic' or 'openai'")
        from .utils.local_graph import sqlite_fts5_error
        fts5_error = sqlite_fts5_error()
        if fts5_error:
            errors.append(fts5_error)
        if cls.DEBUG:
            import warnings
            warnings.warn("Flask DEBUG mode is enabled. Do not use in production.", RuntimeWarning)
        return errors


# ---------------------------------------------------------------------------
# Per-stage model selection (single source of truth, also used by the
# simulation scripts in backend/scripts/).
# ---------------------------------------------------------------------------

LLM_STAGES = ("graph", "simulation", "report")

STAGE_MODEL_ENV_VARS = {
    "graph": "LLM_MODEL_GRAPH",
    "simulation": "LLM_MODEL_SIMULATION",
    "report": "LLM_MODEL_REPORT",
}

# Built-in defaults when neither the stage variable nor LLM_MODEL_NAME is set.
ANTHROPIC_STAGE_DEFAULT_MODELS = {
    "graph": "claude-sonnet-5-5",
    "simulation": "claude-haiku-5-5",
    "report": "claude-opus-5-5",
}
# The "openai" provider is any compatible server, so there is no per-stage
# default: every stage keeps the upstream fallback.
OPENAI_DEFAULT_MODEL = "gpt-4o-mini"


def resolve_stage_model(
    stage: str,
    provider: str | None = None,
    default: str | None = None,
) -> str:
    """Return the model id for one pipeline stage.

    Precedence:
      1. the stage variable (LLM_MODEL_GRAPH / _SIMULATION / _REPORT), if non-empty;
      2. LLM_MODEL_NAME, if non-empty (one model for every stage);
      3. ``default`` if given and non-empty, else the provider's built-in
         default for the stage.

    Variables are read from ``os.environ`` at call time. ``provider`` defaults
    to ``Config.LLM_PROVIDER``.
    """

    if stage not in STAGE_MODEL_ENV_VARS:
        raise ValueError(f"Unknown LLM stage: {stage!r} (expected one of {LLM_STAGES})")

    stage_model = os.environ.get(STAGE_MODEL_ENV_VARS[stage], "").strip()
    if stage_model:
        return stage_model

    shared_model = os.environ.get("LLM_MODEL_NAME", "").strip()
    if shared_model:
        return shared_model

    if default and default.strip():
        return default.strip()

    provider = (provider or Config.LLM_PROVIDER).strip().lower()
    if provider == "anthropic":
        return ANTHROPIC_STAGE_DEFAULT_MODELS[stage]
    return OPENAI_DEFAULT_MODEL
