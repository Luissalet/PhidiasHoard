"""Stdio MCP bridge for Phidias's Hoard.

Every tool call is proxied to the running app (`POST /api/agent/call`) with the Bearer token from
`<DATA_DIR>/mcp-token`; the tool list comes from `GET /api/agent/tools`. Pictures (`material_look`, `model_look`)
arrive as MCP image content. When nothing answers, the bridge starts the app (`python -m phidias_hoard`) and waits for
it; PHIDIAS_BRIDGE_AUTOSTART=0 turns that off."""

from __future__ import annotations

import sys

from phidias_hoard.hoard_link.bridge import CatalogBridge


def main() -> int:
    CatalogBridge(app="phidias", service="phidias-hoard", package="phidias_hoard", default_port=5251, data_dir_env="PHIDIAS_DATA_DIR",
                  title="Phidias's Hoard", root=__file__, default_timeout=180.0, image_content=True).run_bridge()
    return 0


if __name__ == "__main__":
    sys.exit(main())
