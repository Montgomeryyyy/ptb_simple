import re

import numpy as np
from xgboost import XGBClassifier


class XGBModel:
    def __init__(self, params: dict):
        self.params = dict(params)

        n_jobs = int(self.params.pop("n_jobs", 1))
        # Tabular pipeline passes CPU numpy arrays; CPU avoids host/device mismatch warnings.
        self.device = str(self.params.pop("device", "cpu"))
        if "eval_metric" not in self.params:
            self.params["eval_metric"] = "logloss"

        self.model = XGBClassifier(
            tree_method="hist",
            device=self.device,
            n_jobs=n_jobs,
            **self.params,
        )

    def _X(self, X: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(X, dtype=np.float32)

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(self._X(X), y)

    def predict(self, X: np.ndarray):
        return self.model.predict(self._X(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(self._X(X))[:, 1]

    def predict_margin(self, X: np.ndarray) -> np.ndarray:
        """Raw logit before sigmoid (what predict_proba transforms into ehr_pred)."""
        import xgboost as xgb

        dmat = xgb.DMatrix(self._X(X))
        return self.model.get_booster().predict(dmat, output_margin=True)

    def predict_leaf(self, X: np.ndarray) -> np.ndarray:
        """Leaf index per tree — XGBoost's internal sample encoding."""
        import xgboost as xgb

        dmat = xgb.DMatrix(self._X(X))
        leaves = self.model.get_booster().predict(dmat, pred_leaf=True)
        return np.asarray(leaves, dtype=np.int32).reshape(len(X), -1)

    def xgb_score_key_to_column_name(self, key: str, feature_names: list[str]) -> str:
        """Map default XGBoost keys f0..f{n} (numpy fit) to the matching column name."""
        m = re.fullmatch(r"f(\d+)", key)
        if not m:
            return key
        i = int(m.group(1))
        if 0 <= i < len(feature_names):
            return feature_names[i]
        return key

    def get_important_features(self, feature_names: list[str], top_n: int = 10) -> list[str]:
        booster = self.model.get_booster()
        score = booster.get_score(importance_type="gain")
        ranked = sorted(score.items(), key=lambda kv: kv[1], reverse=True)

        out: list[str] = []
        seen: set[str] = set()
        for key, _gain in ranked:
            name = self.xgb_score_key_to_column_name(key, feature_names)
            if name in seen:
                continue
            seen.add(name)
            out.append(name)
            if len(out) >= top_n:
                break

        for name in feature_names:
            if len(out) >= top_n:
                break
            if name not in seen:
                out.append(name)
                seen.add(name)

        return out[:top_n]

    def get_feature_importances_gain(self, feature_names: list[str] | None = None) -> dict[str, float]:
        booster = self.model.get_booster()
        score = booster.get_score(importance_type="gain")
        if feature_names is None:
            return {k: float(v) for k, v in score.items()}
        return {
            self.xgb_score_key_to_column_name(k, feature_names): float(v) for k, v in score.items()
        }
