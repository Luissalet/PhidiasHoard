"""Running the Blender build and reading back what it made."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

SCRIPT = Path(__file__).with_name("blender_build.py")


def has_bpy() -> bool:
    import importlib.util
    return importlib.util.find_spec("bpy") is not None


def run(plan: dict[str, Any], out_dir: Path, blender: Optional[str], timeout_s: float = 1800) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    plan_path = out_dir / "plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    for old in ("remodel.obj", "remodel.blend", "remodel.json"):
        (out_dir / old).unlink(missing_ok=True)
    if blender:
        cmd = [blender, "-b", "--factory-startup", "--python", str(SCRIPT), "--", str(plan_path), str(out_dir)]
    elif has_bpy():
        cmd = [sys.executable, str(SCRIPT), "--", str(plan_path), str(out_dir)]
    else:
        raise RuntimeError("Blender is not installed")
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    marks = [line for line in (proc.stdout or "").splitlines() if line.startswith("PHIDIAS ")]
    if not (out_dir / "remodel.obj").is_file():
        tail = "\n".join(((proc.stdout or "") + "\n" + (proc.stderr or "")).strip().splitlines()[-12:])
        raise RuntimeError(f"Blender did not write the model.\n{tail}")
    stats = json.loads((out_dir / "remodel.json").read_text(encoding="utf-8"))
    stats["log"] = marks
    return stats
