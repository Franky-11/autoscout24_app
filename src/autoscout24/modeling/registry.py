import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib

from autoscout24.config import PROJECT_ROOT

RUNS_DIR = PROJECT_ROOT / "models" / "runs"


@dataclass(frozen=True)
class PersistedRunArtifacts:
    run_id: str
    run_dir: Path
    metadata_path: Path
    pipeline_path: Path


@dataclass(frozen=True)
class LoadedPersistedRun:
    run_id: str
    run_dir: Path
    metadata_path: Path
    pipeline_path: Path
    metadata: dict[str, object]
    pipeline: object


def persist_run(
    pipeline: object,
    metadata: dict[str, object],
    run_id: str | None = None,
) -> PersistedRunArtifacts:
    resolved_run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = RUNS_DIR / resolved_run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = run_dir / "metadata.json"
    pipeline_path = run_dir / "pipeline.joblib"

    joblib.dump(pipeline, pipeline_path)
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=True, indent=2, default=_json_default),
        encoding="utf-8",
    )

    return PersistedRunArtifacts(
        run_id=resolved_run_id,
        run_dir=run_dir,
        metadata_path=metadata_path,
        pipeline_path=pipeline_path,
    )


def load_persisted_runs() -> list[LoadedPersistedRun]:
    if not RUNS_DIR.exists():
        return []

    runs: list[LoadedPersistedRun] = []
    run_directories = sorted(
        (path for path in RUNS_DIR.iterdir() if path.is_dir()),
        key=lambda path: path.name,
    )
    for run_dir in run_directories:
        metadata_path = run_dir / "metadata.json"
        pipeline_path = run_dir / "pipeline.joblib"
        if not metadata_path.exists() or not pipeline_path.exists():
            continue

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        runs.append(
            LoadedPersistedRun(
                run_id=run_dir.name,
                run_dir=run_dir,
                metadata_path=metadata_path,
                pipeline_path=pipeline_path,
                metadata=metadata,
                pipeline=joblib.load(pipeline_path),
            )
        )

    return runs


def _json_default(value: object) -> object:
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")
