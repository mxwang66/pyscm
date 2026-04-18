from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DecisionStump:
    feature_idx: int
    threshold: int
    kind: str = "greater"

    def classify(self, X: np.ndarray) -> np.ndarray:
        return self.classify_feature_values(X[:, self.feature_idx])

    def classify_feature_values(self, feature_values: np.ndarray) -> np.ndarray:
        if self.kind == "greater":
            return feature_values > self.threshold
        return feature_values <= self.threshold

    def inverse(self) -> "DecisionStump":
        return DecisionStump(
            feature_idx=self.feature_idx,
            threshold=self.threshold,
            kind="greater" if self.kind == "less_equal" else "less_equal",
        )

    def __str__(self) -> str:
        operator = ">" if self.kind == "greater" else "<="
        return f"X[{self.feature_idx}] {operator} {self.threshold}"
