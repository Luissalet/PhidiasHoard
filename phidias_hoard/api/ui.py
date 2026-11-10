"""Routes for the bundled studio. Whatever the buttons do is also an agent tool (``POST /api/ui/call`` runs those
handlers); the rest are the files the page shows: meshes for the 3-D view, pictures and the STL download."""

from __future__ import annotations

from typing import Any, Optional
from urllib.parse import quote

from fastapi import APIRouter, Request
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field

from ..agent_tools import TOOLS_BY_NAME
from ..errors import PhidiasError
from .deps import services, tool

router = APIRouter()
NO_CACHE = {"Cache-Control": "no-cache", "X-Content-Type-Options": "nosniff"}


class CallBody(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    arguments: Optional[dict[str, Any]] = None


@router.post("/api/ui/call")
def ui_call(request: Request, body: CallBody):
    if body.name not in TOOLS_BY_NAME:
        raise PhidiasError("not_found", f"No existe la herramienta {body.name}.")
    result = tool(request, body.name, body.arguments or {})
    if isinstance(result, dict):
        result.pop("_image", None)
    return result


@router.get("/api/status")
def status(request: Request):
    return services(request).status()


@router.get("/api/projects/{pid}/mesh/{which}.bin")
def mesh(request: Request, pid: str, which: str):
    if which not in ("before", "sculpt", "closed"):
        raise PhidiasError("not_found", "No existe esa malla.")
    path = services(request).store.dir(pid) / f"{which}.bin"
    if not path.is_file():
        raise PhidiasError("not_found", "Esa malla todavía no existe.")
    return FileResponse(path, media_type="application/octet-stream", headers=NO_CACHE)


@router.get("/api/projects/{pid}/picture/{which}.png")
def picture(request: Request, pid: str, which: str):
    if which not in ("before", "sculpt", "closed"):
        raise PhidiasError("not_found", "No existe esa imagen.")
    path = services(request).store.dir(pid) / f"{which}.png"
    if not path.is_file():
        raise PhidiasError("not_found", "Esa imagen todavía no existe.")
    return FileResponse(path, media_type="image/png", headers=NO_CACHE)


@router.get("/api/projects/{pid}/materials/thumb")
def material_thumb(request: Request, pid: str, name: str):
    return Response(services(request).material_thumb(pid, name), media_type="image/png", headers={"Cache-Control": "max-age=3600"})


@router.get("/api/projects/{pid}/materials/sheet")
def material_sheet(request: Request, pid: str, name: str, recipe: Optional[str] = None):
    return Response(services(request).material_sheet(pid, name, recipe), media_type="image/png", headers=NO_CACHE)


@router.get("/api/projects/{pid}/look.png")
def look(request: Request, pid: str, result: str = "sculpt", view: str = "iso", yaw: Optional[float] = None,
         pitch: Optional[float] = None, size: int = 900, compare: bool = False):
    data = services(request).look(pid, result, view, yaw, pitch, size, None, compare)
    return Response(data, media_type="image/png", headers=NO_CACHE)


@router.get("/api/projects/{pid}/download/{which}.stl")
def download(request: Request, pid: str, which: str):
    svc = services(request)
    project = svc.store.get(pid)
    path = svc.store.dir(pid) / f"{which}.stl"
    if which not in ("sculpt", "closed") or not path.is_file():
        raise PhidiasError("not_found", "Ese resultado todavía no existe.")
    size = int(round(float(project["settings"].get("size_mm", 0))))
    name = f"{project['name']} {size}mm{' cerrado' if which == 'closed' else ''}.stl"
    return FileResponse(path, media_type="model/stl",
                        headers={"Content-Disposition": f"attachment; filename*=UTF-8''{quote(name)}", **NO_CACHE})
