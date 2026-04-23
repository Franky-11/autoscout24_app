from dataclasses import asdict, dataclass

from autoscout24.modeling.service import DEFAULT_BASE_FEATURES, MODEL_LABELS


@dataclass(frozen=True)
class FeatureSelectionConfig:
    base_features: list[str]
    engineered_features: list[str]

    @property
    def all_features(self) -> list[str]:
        return [*self.base_features, *self.engineered_features]

    def to_dict(self) -> dict[str, list[str]]:
        return asdict(self)


@dataclass(frozen=True)
class TrainingConfig:
    model_key: str
    scaler_key: str
    pca_enabled: bool
    n_components: int
    test_size: float
    target_transform: str = "raw"
    cv_folds: int = 3

    @property
    def model_label(self) -> str:
        return MODEL_LABELS[self.model_key]

    @property
    def target_transform_label(self) -> str:
        return {"raw": "Preis direkt", "log1p": "log1p(Preis)"}[self.target_transform]

    @property
    def effective_pca_components(self) -> int | str:
        return self.n_components if self.pca_enabled else "N/A"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


DEFAULT_FEATURE_SELECTION = FeatureSelectionConfig(
    base_features=DEFAULT_BASE_FEATURES,
    engineered_features=[],
)
