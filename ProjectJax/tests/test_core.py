from jax.core import Jax, ToolResult
from jax.tools import BaseTool, ToolRegistry

class FakeTool(BaseTool):
    name = "fake"
    description = "A test tool."
    mutates = False

    @property
    def parameters(self):
        return {"type": "object", "properties": {}}

    def execute(self, **kwargs):
        return ToolResult(True, {"answer": 42})

class FakeModel:
    def __init__(self):
        self.calls = 0

    def chat(self, messages, tools, *, think=True):
        self.calls += 1
        if self.calls == 1:
            return {"message": {"role": "assistant", "tool_calls": [
                {"function": {"name": "fake", "arguments": {}}}
            ]}}
        return {"message": {"role": "assistant", "content": "done"}}

    @staticmethod
    def encode_tool_result(result):
        return str(result.data)

class FakeLogger:
    def record(self, *args):
        pass

class AllowPolicy:
    class Decision:
        allowed = True
        reason = "allowed"

    def check(self, tool, args):
        return self.Decision()

    def confirm(self, tool, args):
        return True

def test_tool_loop():
    j = Jax(FakeModel(), ToolRegistry([FakeTool()]), AllowPolicy(), FakeLogger())
    assert j.run("do it") == "done"


def test_model_failure_marks_task_failed():
    class FailingModel(FakeModel):
        def chat(self, messages, tools, *, think=True):
            raise RuntimeError("model unavailable")

    j = Jax(FailingModel(), ToolRegistry([FakeTool()]), AllowPolicy(), FakeLogger())
    try:
        j.run("do it")
    except RuntimeError:
        pass
    else:
        raise AssertionError("model failure should propagate")
    task = next(iter(j.tasks.tasks.values()))
    assert task.status.value == "failed"
    assert "model unavailable" in task.error


def test_thinking_command_persists_across_requests():
    class RecordingModel:
        def __init__(self):
            self.thinks = []

        def chat(self, messages, tools, *, think=True):
            self.thinks.append(think)
            return {"message": {"role": "assistant", "content": "ok"}}

        @staticmethod
        def encode_tool_result(result):
            return str(result.data)

    model = RecordingModel()
    j = Jax(model, ToolRegistry([FakeTool()]), AllowPolicy(), FakeLogger())

    assert j.run("stop thinking") == "Thinking disabled."
    assert j.run("first request") == "ok"
    assert j.run("second request") == "ok"
    assert model.thinks == [False, False]

    assert j.run("start thinking") == "Thinking enabled."
    assert j.run("third request") == "ok"
    assert model.thinks == [False, False, True]


def test_thinking_commands_are_case_insensitive():
    class RecordingModel:
        def chat(self, messages, tools, *, think=True):
            raise AssertionError("thinking command should not call the model")

        @staticmethod
        def encode_tool_result(result):
            return str(result.data)

    j = Jax(RecordingModel(), ToolRegistry([FakeTool()]), AllowPolicy(), FakeLogger())
    assert j.run(" STOP THINKING ") == "Thinking disabled."
    assert j.thinking_enabled is False
    assert j.run(" THINKING ON ") == "Thinking enabled."
    assert j.thinking_enabled is True


def test_system_prompt_defines_authorized_root_and_relative_path_semantics(tmp_path):
    class RecordingModel:
        def __init__(self):
            self.messages = None

        def chat(self, messages, tools, *, think=True):
            self.messages = messages
            return {"message": {"role": "assistant", "content": "ok"}}

        @staticmethod
        def encode_tool_result(result):
            return str(result.data)

    model = RecordingModel()
    from jax.policy import Policy
    j = Jax(model, ToolRegistry([FakeTool()]), Policy(tmp_path), FakeLogger())
    assert j.run("inspect unsorted") == "ok"
    prompt = model.messages[0]["content"]
    assert str(tmp_path) in prompt
    assert "relative path such as 'unsorted'" in prompt
    assert "Do not turn a relative path into an absolute path" in prompt


class MutatingFakeTool(BaseTool):
    name = "mutating_fake"
    description = "A mutating test tool."
    mutates = True

    def __init__(self):
        self.executions = 0

    @property
    def parameters(self):
        return {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        }

    def execute(self, **kwargs):
        self.executions += 1
        return ToolResult(True, {"executed": self.executions, "path": kwargs["path"]})


def test_successful_mutation_may_retry_on_a_later_model_turn():
    class RepeatingModel:
        def __init__(self):
            self.calls = 0
            self.tool_messages = []

        def chat(self, messages, tools, *, think=True):
            self.calls += 1
            self.tool_messages = [m for m in messages if m.get("role") == "tool"]
            if self.calls < 3:
                return {
                    "message": {
                        "role": "assistant",
                        "tool_calls": [
                            {"function": {"name": "mutating_fake", "arguments": {"path": "a.txt"}}}
                        ],
                    }
                }
            return {"message": {"role": "assistant", "content": "done"}}

        @staticmethod
        def encode_tool_result(result):
            return str(result.data)

    tool = MutatingFakeTool()
    model = RepeatingModel()
    j = Jax(model, ToolRegistry([tool]), AllowPolicy(), FakeLogger())

    assert j.run("do it") == "done"
    assert tool.executions == 2


def test_identical_successful_mutations_in_one_model_response_execute_once():
    class RepeatingBatchModel:
        def __init__(self):
            self.calls = 0

        def chat(self, messages, tools, *, think=True):
            self.calls += 1
            if self.calls == 1:
                call = {"function": {"name": "mutating_fake", "arguments": {"path": "a.txt"}}}
                return {"message": {"role": "assistant", "tool_calls": [call, call]}}
            return {"message": {"role": "assistant", "content": "done"}}

        @staticmethod
        def encode_tool_result(result):
            return str(result.data)

    tool = MutatingFakeTool()
    j = Jax(RepeatingBatchModel(), ToolRegistry([tool]), AllowPolicy(), FakeLogger())

    assert j.run("do it") == "done"
    assert tool.executions == 1


def test_filesystem_task_survives_repeated_create_and_move_requests(tmp_path):
    from jax.logging import OperationLogger
    from jax.policy import Policy
    from jax.tools import CreateDirectory, MoveFile

    (tmp_path / "unsorted").mkdir()
    (tmp_path / "unsorted" / "data.csv").write_text("a,b\n1,2\n")

    class RepeatingOrganizerModel:
        def __init__(self):
            self.calls = 0

        def chat(self, messages, tools, *, think=True):
            self.calls += 1
            calls = {
                1: ("create_directory", {"path": "unsorted/csv"}),
                2: ("create_directory", {"path": "unsorted/csv"}),
                3: ("move_file", {"source": "unsorted/data.csv", "destination": "unsorted/csv/data.csv"}),
                4: ("move_file", {"source": "unsorted/data.csv", "destination": "unsorted/csv/data.csv"}),
            }
            if self.calls in calls:
                name, arguments = calls[self.calls]
                return {
                    "message": {
                        "role": "assistant",
                        "tool_calls": [{"function": {"name": name, "arguments": arguments}}],
                    }
                }
            return {
                "message": {
                    "role": "assistant",
                    "content": "Moved data.csv to unsorted/csv/data.csv and completed the task.",
                }
            }

        @staticmethod
        def encode_tool_result(result):
            return str({"ok": result.ok, "data": result.data, "error": result.error})

    model = RepeatingOrganizerModel()
    jax = Jax(
        model,
        ToolRegistry([CreateDirectory(tmp_path), MoveFile(tmp_path)]),
        Policy(tmp_path, input_fn=lambda _: "y"),
        OperationLogger(tmp_path / "ops.jsonl"),
        max_turns=6,
    )

    result = jax.run("Organize data.csv into a csv directory. Do not modify anything outside the sandbox.")

    assert "completed" in result
    assert not (tmp_path / "unsorted" / "data.csv").exists()
    assert (tmp_path / "unsorted" / "csv" / "data.csv").read_text() == "a,b\n1,2\n"
    assert model.calls == 5
