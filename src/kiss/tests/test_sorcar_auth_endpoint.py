import uvicorn
from starlette.testclient import TestClient

from kiss.agents.sorcar import sorcar as sorcar_module


class _CapturingServer:
    captured_app = None

    def __init__(self, config: uvicorn.Config) -> None:
        self.config = config
        self.handle_exit = lambda sig, frame: None
        type(self).captured_app = config.app

    def run(self) -> None:
        return None


def test_auth_endpoint_returns_status_payload(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(sorcar_module, "_KISS_DIR", tmp_path / ".kiss")
    monkeypatch.setattr(sorcar_module, "_cleanup_stale_cs_dirs", lambda: None)
    monkeypatch.setattr(sorcar_module, "_restore_merge_files", lambda *args, **kwargs: None)
    monkeypatch.setattr(sorcar_module, "_scan_files", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(sorcar_module, "_load_last_model", lambda: "")
    monkeypatch.setattr(sorcar_module.shutil, "which", lambda _name: None)
    monkeypatch.setattr(sorcar_module.webbrowser, "open", lambda _url: True)
    monkeypatch.setattr(sorcar_module, "find_free_port", lambda: 18765)
    monkeypatch.setattr(uvicorn, "Server", _CapturingServer)

    _CapturingServer.captured_app = None

    sorcar_module.run_chatbot(
        agent_factory=lambda _name: None,
        title="Test Sorcar",
        work_dir=str(tmp_path),
    )

    assert _CapturingServer.captured_app is not None

    client = TestClient(_CapturingServer.captured_app)
    response = client.get("/auth?model=gpt-5.4")

    assert response.status_code == 200
    payload = response.json()
    assert payload["model"] == "gpt-5.4"
    assert payload["is_openai_model"] is True
    assert "preferred_auth" in payload
