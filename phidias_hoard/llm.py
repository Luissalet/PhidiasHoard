"""The model that looks at materials and at sculpts. By default it is whatever the family resolves for vision through
Hoard Link (the model served on the Sparks); settings ``llm_url`` / ``llm_model`` point at one OpenAI-compatible server
instead. Pictures always go as images."""

from __future__ import annotations

import base64
import json
import re
import threading
from typing import Any, Optional

import httpx

from .errors import PhidiasError


def extract_json(text: str) -> dict[str, Any]:
    """The JSON object in an answer, whether it comes bare, fenced or after some thinking."""
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S)
    candidates = fenced[::-1]
    start = text.find("{")
    if start >= 0:
        candidates.append(text[start: text.rfind("}") + 1])
    for cand in candidates:
        try:
            data = json.loads(cand)
            if isinstance(data, dict):
                return data
        except ValueError:
            continue
    raise PhidiasError("failed", "El modelo no devolvió un JSON que se pueda leer.", "Vuelve a lanzarlo; si se repite, prueba con otro modelo en Ajustes.")


class ModelClient:
    def __init__(self, settings: Any):
        self.settings = settings
        self._link = None
        self._lock = threading.Lock()

    def endpoint(self) -> tuple[str, str]:
        url, model = self.settings["llm_url"], self.settings["llm_model"]
        if url:
            base = url.rstrip("/")
            if not base.endswith("/chat/completions"):
                base += "/chat/completions" if base.endswith("/v1") else "/v1/chat/completions"
            return base, model
        try:
            with self._lock:
                if self._link is None:
                    from .hoard_link import Link, LinkConfig
                    self._link = Link(LinkConfig.load(app="phidias"))
            res = self._link.sync.resolve("vision")
        except Exception as exc:  # noqa: BLE001
            raise PhidiasError("unavailable", f"No hay modelo disponible: {type(exc).__name__}: {exc}"[:300],
                               "Comprueba que Faustus y las Sparks están en marcha, o indica un servidor en Ajustes.") from None
        if getattr(res, "state", "") != "resolved" or not getattr(res, "url", ""):
            raise PhidiasError("unavailable", f"No hay modelo con visión disponible: {getattr(res, 'reason', '')}"[:300],
                               "Comprueba que Faustus y las Sparks están en marcha, o indica un servidor en Ajustes.")
        return res.url, res.model or ""

    def status(self) -> dict[str, Any]:
        try:
            url, model = self.endpoint()
            return {"available": True, "url": url, "model": model or "(el del servidor)"}
        except PhidiasError as exc:
            return {"available": False, "reason": exc.message}

    def ask(self, system: str, user: str, images: list[bytes], max_tokens: int = 6000, timeout: Optional[float] = None) -> dict[str, Any]:
        """One question with pictures; the answer's JSON object."""
        url, model = self.endpoint()
        content: list[dict[str, Any]] = [{"type": "text", "text": user}]
        for img in images:
            content.append({"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(img).decode()}})
        body: dict[str, Any] = {"messages": [{"role": "system", "content": system}, {"role": "user", "content": content}],
                                "max_tokens": max_tokens, "temperature": 0.2, "stream": False}
        if model:
            body["model"] = model
        try:
            r = httpx.post(url, json=body, timeout=timeout or float(self.settings["llm_timeout_s"]))
            r.raise_for_status()
            msg = r.json()["choices"][0]["message"]
        except PhidiasError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise PhidiasError("unavailable", f"El modelo no contestó: {type(exc).__name__}: {exc}"[:300]) from None
        text = msg.get("content") or ""
        if isinstance(text, list):
            text = "".join(part.get("text", "") for part in text if isinstance(part, dict))
        return extract_json(text or msg.get("reasoning_content") or "")
