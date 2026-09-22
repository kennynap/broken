import json


def test_ollama_model_records_metrics(monkeypatch):
    import jax.model as model_module

    class Response:
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc, tb):
            return False
        def read(self):
            return json.dumps({
                "message": {"role": "assistant", "content": "hello"},
                "load_duration": 100_000_000,
                "prompt_eval_count": 20,
                "prompt_eval_duration": 200_000_000,
                "eval_count": 40,
                "eval_duration": 2_000_000_000,
            }).encode()

    captured = {}
    def fake_urlopen(request, timeout):
        captured["payload"] = json.loads(request.data)
        return Response()
    monkeypatch.setattr(model_module, "urlopen", fake_urlopen)
    client = model_module.OllamaModel("http://localhost:11434", "qwen3:8b")

    result = client.chat([{"role": "user", "content": "hello"}], [], think=False)
    assert captured["payload"]["think"] is False

    assert result["message"]["content"] == "hello"
    metrics = client.last_metrics
    assert metrics is not None
    assert metrics.total_ms >= 0
    assert metrics.response_decode_ms >= 0
    assert metrics.load_ms == 100.0
    assert metrics.prompt_eval_ms == 200.0
    assert metrics.prompt_eval_tokens == 20
    assert metrics.eval_ms == 2000.0
    assert metrics.eval_tokens == 40
    assert metrics.eval_tokens_per_second == 20.0
    assert metrics.thinking is False
    assert client.request_count == 1
    assert client.metrics_history == [metrics]


def test_ollama_model_handles_missing_metrics(monkeypatch):
    import jax.model as model_module

    class Response:
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc, tb):
            return False
        def read(self):
            return b'{"message":{"role":"assistant","content":"ok"}}'

    monkeypatch.setattr(model_module, "urlopen", lambda request, timeout: Response())
    client = model_module.OllamaModel("http://localhost:11434", "qwen3:8b")

    client.chat([{"role": "user", "content": "hello"}], [])

    metrics = client.last_metrics
    assert metrics is not None
    assert metrics.load_ms is None
    assert metrics.prompt_eval_ms is None
    assert metrics.prompt_eval_tokens is None
    assert metrics.eval_ms is None
    assert metrics.eval_tokens is None
    assert metrics.eval_tokens_per_second is None
    assert metrics.thinking is False


def test_ollama_model_detects_thinking(monkeypatch):
    import jax.model as model_module

    class Response:
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc, tb):
            return False
        def read(self):
            return json.dumps({
                "message": {"role": "assistant", "thinking": "internal", "content": "hello"},
                "eval_count": 1,
                "eval_duration": 1_000_000,
            }).encode()

    monkeypatch.setattr(model_module, "urlopen", lambda request, timeout: Response())
    client = model_module.OllamaModel("http://localhost:11434", "qwen3:8b")
    client.chat([{"role": "user", "content": "hello"}], [])
    assert client.last_metrics is not None
    assert client.last_metrics.thinking is True


def test_format_metrics_is_compact():
    import jax.model as model_module
    metrics = model_module.ModelMetrics(
        total_ms=2840.0, response_decode_ms=0.5, load_ms=120.0,
        prompt_eval_ms=80.0, prompt_eval_tokens=10, eval_ms=2610.0,
        eval_tokens=82, eval_tokens_per_second=31.4, thinking=False,
    )
    output = model_module.format_metrics(metrics, 1)
    assert output == "[Model: 2.84s | load: 0.12s | prompt: 0.08s | generate: 2.61s | tokens: 82 | speed: 31.4 tok/s | calls: 1 | thinking: no]"


def test_ollama_model_sends_thinking_setting(monkeypatch):
    import jax.model as model_module

    class Response:
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc, tb):
            return False
        def read(self):
            return b'{"message":{"role":"assistant","content":"ok"}}'

    captured = []
    def fake_urlopen(request, timeout):
        captured.append(json.loads(request.data)["think"])
        return Response()

    monkeypatch.setattr(model_module, "urlopen", fake_urlopen)
    client = model_module.OllamaModel("http://localhost:11434", "qwen3:8b")
    client.chat([{"role": "user", "content": "hello"}], [], think=False)
    client.chat([{"role": "user", "content": "hello"}], [], think=True)
    assert captured == [False, True]
