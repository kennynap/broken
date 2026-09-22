import pytest


def test_config_reads_environment_per_instance(monkeypatch, tmp_path):
    from jax.config import Config

    monkeypatch.setenv("JAX_ALLOWED_ROOT", str(tmp_path))
    monkeypatch.setenv("JAX_DEBUG", "true")
    config = Config()

    assert config.allowed_root == tmp_path.resolve()
    assert config.debug is True


def test_build_jax_uses_supplied_root(tmp_path):
    from jax.config import Config
    from jax.main import build_jax

    jax = build_jax(Config(allowed_root=tmp_path))

    assert jax.policy.authorized_root == tmp_path.resolve()
    assert "write_text_file" in jax.registry._tools
    assert "move_file" in jax.registry._tools
    assert "run_command" in jax.registry._tools


def test_main_debug_path_uses_existing_config(monkeypatch, capsys, tmp_path):
    import jax.main as main_module

    monkeypatch.setenv("JAX_ALLOWED_ROOT", str(tmp_path))
    monkeypatch.setenv("JAX_DEBUG", "true")

    class FakeModel:
        request_count = 0
        last_metrics = None

    class FakeJax:
        model = FakeModel()
        registry = type("Registry", (), {"_tools": {"write_text_file": type("Tool", (), {"name": "write_text_file"})()}})()

        def run(self, request):
            return "done"

    monkeypatch.setattr(main_module, "build_jax", lambda config: FakeJax())
    inputs = iter(["hello", "exit"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))

    main_module.main()
    output = capsys.readouterr().out

    assert f"Authorized root: {tmp_path.resolve()}" in output
    assert "[Debug tools: write_text_file]" in output
