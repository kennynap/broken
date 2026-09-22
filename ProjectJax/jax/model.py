import json
import time
from dataclasses import dataclass
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class ModelMetrics:
    total_ms: float
    response_decode_ms: float
    load_ms: float | None
    prompt_eval_ms: float | None
    prompt_eval_tokens: int | None
    eval_ms: float | None
    eval_tokens: int | None
    eval_tokens_per_second: float | None
    thinking: bool


class OllamaModel:
    def __init__(self, base_url: str, model: str):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.last_metrics: ModelMetrics | None = None
        self.metrics_history: list[ModelMetrics] = []
        self.request_count = 0

    def chat(self, messages, tools, *, think=True):
        self.request_count += 1
        payload = {
            "model": self.model,
            "messages": messages,
            "tools": tools,
            "stream": False,
            "think": think,
        }
        request = Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        started_ns = time.perf_counter_ns()
        with urlopen(request, timeout=120) as response:
            body = response.read()
        decode_started_ns = time.perf_counter_ns()
        result = json.loads(body)
        finished_ns = time.perf_counter_ns()
        self.last_metrics = _metrics_from_response(
            result,
            total_ms=(finished_ns - started_ns) / 1_000_000,
            response_decode_ms=(finished_ns - decode_started_ns) / 1_000_000,
        )
        self.metrics_history.append(self.last_metrics)
        return result

    @staticmethod
    def encode_tool_result(result):
        return json.dumps({
            "ok": result.ok,
            "data": result.data,
            "error": result.error,
        }, default=str)


def _duration_ms(value):
    if not isinstance(value, (int, float)):
        return None
    return value / 1_000_000


def _metrics_from_response(result, total_ms, response_decode_ms):
    eval_ns = result.get("eval_duration")
    eval_tokens = result.get("eval_count")
    eval_ms = _duration_ms(eval_ns)
    tokens_per_second = None
    if isinstance(eval_tokens, int) and isinstance(eval_ns, (int, float)) and eval_ns > 0:
        tokens_per_second = eval_tokens / (eval_ns / 1_000_000_000)
    return ModelMetrics(
        total_ms=total_ms,
        response_decode_ms=response_decode_ms,
        load_ms=_duration_ms(result.get("load_duration")),
        prompt_eval_ms=_duration_ms(result.get("prompt_eval_duration")),
        prompt_eval_tokens=result.get("prompt_eval_count") if isinstance(result.get("prompt_eval_count"), int) else None,
        eval_ms=eval_ms,
        eval_tokens=eval_tokens if isinstance(eval_tokens, int) else None,
        eval_tokens_per_second=tokens_per_second,
        thinking=_response_used_thinking(result),
    )


def _response_used_thinking(result):
    message = result.get("message")
    if not isinstance(message, dict):
        return False
    thinking = message.get("thinking")
    return isinstance(thinking, str) and bool(thinking.strip())


def format_metrics(metrics, calls):
    parts = [f"Model: {metrics.total_ms / 1000:.2f}s"]
    if metrics.load_ms is not None:
        parts.append(f"load: {metrics.load_ms / 1000:.2f}s")
    if metrics.prompt_eval_ms is not None:
        parts.append(f"prompt: {metrics.prompt_eval_ms / 1000:.2f}s")
    if metrics.eval_ms is not None:
        parts.append(f"generate: {metrics.eval_ms / 1000:.2f}s")
    if metrics.eval_tokens is not None:
        parts.append(f"tokens: {metrics.eval_tokens}")
    if metrics.eval_tokens_per_second is not None:
        parts.append(f"speed: {metrics.eval_tokens_per_second:.1f} tok/s")
    parts.append(f"calls: {calls}")
    parts.append(f"thinking: {'yes' if metrics.thinking else 'no'}")
    return "[" + " | ".join(parts) + "]"
