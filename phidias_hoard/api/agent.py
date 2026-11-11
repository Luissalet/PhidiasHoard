"""/api/agent/* - the bridge used by mcp_server.py (Bearer token from <DATA_DIR>/mcp-token)."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request

from ..agent_tools import AGENT_INSTRUCTIONS, call_tool, tool_catalog
from ..config import REPO_ROOT, Config
from ..hoard_link import skills as family_skills
from ..hoard_link.agentkit import make_agent_router
from .deps import services

# The procedures of the family's skill library that this app's work needs (the manifest says which); every agent that
# connects gets them. create_app points the shelf at the running app's data folder.
SHELF = family_skills.Shelf("phidias", Config.from_env().data_dir, family_skills.wants_of(REPO_ROOT / "faustus-plugin.json"))


def _call(name: str, arguments: dict[str, Any], request: Request) -> Any:
    return call_tool(services(request), name, arguments)


router = APIRouter()
router.include_router(make_agent_router(tools_fn=tool_catalog, call_fn=_call, token_fn=lambda request: services(request).token,
                                        instructions=AGENT_INSTRUCTIONS, app_name="phidias", skills=SHELF))

