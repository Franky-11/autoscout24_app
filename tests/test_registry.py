from autoscout24.modeling.registry import persist_run


def test_persist_run_writes_pipeline_and_metadata(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "autoscout24.modeling.registry.RUNS_DIR",
        tmp_path / "runs",
    )

    artifacts = persist_run(
        pipeline={"model": "dummy"},
        metadata={"metric": 1.23},
        run_id="run-test",
    )

    assert artifacts.run_id == "run-test"
    assert artifacts.pipeline_path.exists()
    assert artifacts.metadata_path.exists()
    assert '"metric": 1.23' in artifacts.metadata_path.read_text(encoding="utf-8")
