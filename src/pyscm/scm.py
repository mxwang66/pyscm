import logging

import numpy as np

from ._scm_utility import find_max
from .rules import DecisionStump


class _RuleListModel:
    def __init__(self, model_type: str):
        self.model_type = model_type
        self.rules = []

    def add(self, rule: DecisionStump) -> None:
        self.rules.append(rule)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model_type == "conjunction":
            predictions = np.ones(X.shape[0], dtype=bool)
            for rule in self.rules:
                np.logical_and(predictions, rule.classify(X), out=predictions)
        else:
            predictions = np.zeros(X.shape[0], dtype=bool)
            for rule in self.rules:
                np.logical_or(predictions, rule.classify(X), out=predictions)
        return predictions.astype(np.uint8)

    def __len__(self) -> int:
        return len(self.rules)


class SetCoveringMachineClassifier:
    def __init__(self, p: float = 1.0, model_type: str = "conjunction", max_rules: int = 10):
        self.p = p
        self.model_type = model_type
        self.max_rules = max_rules

    @staticmethod
    def _validate_X_uint8(X: np.ndarray) -> np.ndarray:
        if not isinstance(X, np.ndarray):
            raise TypeError("X must be a numpy.ndarray.")
        if X.dtype != np.uint8:
            raise TypeError("X must have dtype np.uint8.")
        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")
        if not X.flags.f_contiguous:
            raise ValueError("X must be F-contiguous.")
        return X

    def _validate_fit_inputs(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        X = self._validate_X_uint8(X)
        if not isinstance(y, np.ndarray):
            raise TypeError("y must be a numpy.ndarray.")
        if y.dtype != np.uint8:
            raise TypeError("y must have dtype np.uint8.")
        if y.ndim != 1:
            raise ValueError("y must be a 1D array.")
        if not y.flags.c_contiguous:
            raise ValueError("y must be C-contiguous.")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of rows.")

        classes = np.unique(y)
        if not np.array_equal(classes, np.array([0, 1], dtype=np.uint8)):
            raise ValueError("y must contain only binary labels {0, 1}.")

        return X, y

    def _assert_is_fitted(self) -> None:
        if not hasattr(self, "model_"):
            raise RuntimeError("SetCoveringMachineClassifier must be fitted before calling predict().")

    def fit(self, X, y, tiebreaker=None, iteration_callback=None):
        if self.model_type not in {"conjunction", "disjunction"}:
            raise ValueError("Unsupported model type.")

        if iteration_callback is None:
            iteration_callback = lambda _: None

        X, y = self._validate_fit_inputs(X, y)
        Xt = X.T

        if self.model_type == "conjunction":
            pos_ex_idx = np.where(y == 1)[0].astype(np.intp, copy=False)
            neg_ex_idx = np.where(y == 0)[0].astype(np.intp, copy=False)
        else:
            # Learn a conjunction on inverted labels and invert added rules.
            pos_ex_idx = np.where(y == 0)[0].astype(np.intp, copy=False)
            neg_ex_idx = np.where(y == 1)[0].astype(np.intp, copy=False)

        y_unified = np.zeros(len(y), dtype=np.uint8)
        y_unified[pos_ex_idx] = 1

        x_argsort_by_feature_t = np.argsort(X.T, axis=1)
        self.model_ = _RuleListModel(self.model_type)

        remaining_example_idx = np.arange(len(y_unified), dtype=np.intp)
        remaining_negative_example_idx = neg_ex_idx.astype(np.intp, copy=False)

        while len(remaining_negative_example_idx) > 0 and len(self.model_) < self.max_rules:
            (
                opti_utility,
                opti_feat_idx,
                opti_threshold,
                opti_kind,
                opti_n,
                opti_p_bar,
            ) = find_max(self.p, Xt, y_unified, x_argsort_by_feature_t, remaining_example_idx)

            if len(opti_feat_idx) > 1:
                if tiebreaker is None:
                    training_risk_decrease = (1.0 * opti_n) - opti_p_bar
                    keep_idx = np.where(training_risk_decrease == training_risk_decrease.max())[0][0]
                else:
                    keep_idx = tiebreaker(self.model_type, opti_feat_idx, opti_threshold, opti_kind)
            else:
                keep_idx = 0

            added_rule = DecisionStump(
                feature_idx=int(opti_feat_idx[keep_idx]),
                threshold=int(opti_threshold[keep_idx]),
                kind="greater" if int(opti_kind[keep_idx]) == 0 else "less_equal",
            )
            if self.model_type == "disjunction":
                added_rule = added_rule.inverse()
            self.model_.add(added_rule)

            logging.debug("The best rule has utility %.3f", opti_utility)

            feature_values = X[:, added_rule.feature_idx]
            remaining_example_idx = remaining_example_idx[
                added_rule.classify_feature_values(feature_values[remaining_example_idx])
            ]
            remaining_negative_example_idx = remaining_negative_example_idx[
                added_rule.classify_feature_values(feature_values[remaining_negative_example_idx])
            ]

            iteration_callback(self.model_)

        return self

    def predict(self, X):
        self._assert_is_fitted()
        X = self._validate_X_uint8(X)
        return self.model_.predict(X)
