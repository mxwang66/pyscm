import logging
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ._scm_utility import find_max


@dataclass(frozen=True, slots=True)
class SCMRule:
    feature_idx: int
    threshold: int
    kind: str = "greater"

    def classify(self, X: NDArray[np.uint8]) -> NDArray[np.bool_]:
        return self.classify_feature_values(X[:, self.feature_idx])

    def classify_feature_values(self, feature_values: NDArray[np.uint8]) -> NDArray[np.bool_]:
        if self.kind == "greater":
            return feature_values > self.threshold
        return feature_values <= self.threshold

    def inverse(self) -> "SCMRule":
        return SCMRule(
            feature_idx=self.feature_idx,
            threshold=self.threshold,
            kind="greater" if self.kind == "less_equal" else "less_equal",
        )

    def __str__(self) -> str:
        operator = ">" if self.kind == "greater" else "<="
        return f"X[{self.feature_idx}] {operator} {self.threshold}"


@dataclass(frozen=True, slots=True)
class SCMModel:
    model_type: str
    rules: tuple[SCMRule]


def _validate_X(X: NDArray[np.uint8]) -> None:
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy.ndarray.")
    if X.dtype != np.uint8:
        raise TypeError("X must have dtype np.uint8.")
    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")
    if not X.flags.f_contiguous:
        raise ValueError("X must be F-contiguous.")


def _validate_data(X: NDArray[np.uint8], y: NDArray[np.bool_]) -> None:
    _validate_X(X)

    if not isinstance(y, np.ndarray):
        raise TypeError("y must be a numpy.ndarray.")
    if y.dtype != np.bool_:
        raise TypeError("y must have dtype np.bool_.")
    if y.ndim != 1:
        raise ValueError("y must be a 1D array.")
    if not y.flags.c_contiguous:
        raise ValueError("y must be C-contiguous.")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of rows.")

    classes = np.unique(y)
    if not np.array_equal(classes, np.array([False, True], dtype=np.bool_)):
        raise ValueError("y must contain both boolean labels {False, True}.")


def fit_scm(
    X: NDArray[np.uint8],
    y: NDArray[np.bool_],
    p: float = 1.0,
    model_type: str = "conjunction",
    max_rules: int = 10,
) -> SCMModel:
    if model_type not in {"conjunction", "disjunction"}:
        raise ValueError("Unsupported model type.")

    _validate_data(X, y)
    Xt = X.T
    remaining_idx = np.arange(len(y), dtype=np.intp)

    if model_type == "conjunction":
        remaining_neg_idx = np.where(~y)[0]
        y_unified = y.astype(np.uint8, copy=False)
    else:
        # Learn a conjunction on inverted labels and inverted rules.
        remaining_neg_idx = np.where(y)[0]
        y_unified = (~y).astype(np.uint8, copy=False)

    rules: list[SCMRule] = []

    while len(remaining_neg_idx) > 0 and len(rules) < max_rules:
        (
            opti_utility,
            opti_feat_idx,
            opti_threshold,
            opti_kind,
            opti_n,
            opti_p_bar,
        ) = find_max(p, Xt, y_unified, remaining_idx)

        # Tiebreaker
        if len(opti_feat_idx) > 1:
            training_risk_decrease = (1.0 * opti_n) - opti_p_bar
            keep_idx = np.where(training_risk_decrease == training_risk_decrease.max())[0][0]
        else:
            keep_idx = 0

        training_rule = SCMRule(
            feature_idx=int(opti_feat_idx[keep_idx]),
            threshold=int(opti_threshold[keep_idx]),
            kind="greater" if int(opti_kind[keep_idx]) == 0 else "less_equal",
        )
        rules.append(training_rule)

        logging.debug("The best rule has utility %.3f", opti_utility)

        feature_values = X[:, training_rule.feature_idx]
        remaining_idx = remaining_idx[
            training_rule.classify_feature_values(feature_values[remaining_idx])
        ]
        remaining_neg_idx = remaining_neg_idx[
            training_rule.classify_feature_values(feature_values[remaining_neg_idx])
        ]

    # Invert rules for disjunction
    if model_type == "disjunction":
        rules = list(r.inverse() for r in rules)

    return SCMModel(model_type=model_type, rules=tuple(rules))


def predict_scm(model: SCMModel, X: NDArray[np.uint8]) -> NDArray[np.bool_]:
    _validate_X(X)

    if model.model_type == "conjunction":
        predictions = np.ones(X.shape[0], dtype=np.bool_)
        for rule in model.rules:
            np.logical_and(predictions, rule.classify(X), out=predictions)
        return predictions

    if model.model_type == "disjunction":
        predictions = np.zeros(X.shape[0], dtype=np.bool_)
        for rule in model.rules:
            np.logical_or(predictions, rule.classify(X), out=predictions)
        return predictions

    raise ValueError("Unsupported model type.")
