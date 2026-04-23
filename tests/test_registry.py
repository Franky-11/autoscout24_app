from autoscout24.modeling.registry import load_persisted_runs, persist_run


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


def test_load_persisted_runs_reads_metadata_and_pipeline(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "autoscout24.modeling.registry.RUNS_DIR",
        tmp_path / "runs",
    )
    persist_run(
        pipeline={"model": "dummy"},
        metadata={"metric": 1.23},
        run_id="run-test",
    )

    loaded_runs = load_persisted_runs()

    assert len(loaded_runs) == 1
    assert loaded_runs[0].run_id == "run-test"
    assert loaded_runs[0].metadata["metric"] == 1.23
    assert loaded_runs[0].pipeline == {"model": "dummy"}
