import importlib
import sys
from unittest import TestCase

import numpy as np

from pyscm._scm_utility import find_max
from pyscm.rules import DecisionStump
from pyscm.scm import SetCoveringMachineClassifier


class UtilityTests(TestCase):
    def test_find_max_basic(self):
        X = np.asfortranarray(
            np.array([[1, 2, 2, 2, 3, 4]], dtype=np.uint8).reshape(-1, 1).copy()
        )
        y = np.array([0, 1, 0, 1, 1, 1], dtype=np.uint8)
        xas = np.ascontiguousarray(np.argsort(X, axis=0).T, dtype=np.intp)

        best_utility, best_feat_idx, best_thresholds, best_kinds, _, _ = find_max(
            1.0, X.T, y, xas, np.arange(X.shape[0], dtype=np.intp)
        )
        np.testing.assert_almost_equal(actual=best_utility, desired=1.0)
        np.testing.assert_almost_equal(actual=best_feat_idx, desired=[0])
        np.testing.assert_almost_equal(actual=best_thresholds, desired=[1])
        np.testing.assert_almost_equal(actual=best_kinds, desired=[0])

    def test_fit_predict_conjunction(self):
        X = np.asfortranarray(np.array([[0], [1], [2], [3]], dtype=np.uint8))
        y = np.array([0, 0, 1, 1], dtype=np.uint8)

        model = SetCoveringMachineClassifier(model_type="conjunction", max_rules=2)
        model.fit(X, y)
        preds = model.predict(X)

        self.assertEqual(preds.dtype, np.uint8)
        self.assertEqual(preds.shape, y.shape)

    def test_fit_predict_disjunction(self):
        X = np.asfortranarray(np.array([[0], [1], [2], [3]], dtype=np.uint8))
        y = np.array([0, 0, 1, 1], dtype=np.uint8)

        model = SetCoveringMachineClassifier(model_type="disjunction", max_rules=2)
        model.fit(X, y)
        preds = model.predict(X)

        self.assertEqual(preds.dtype, np.uint8)
        self.assertEqual(preds.shape, y.shape)

    def test_predict_before_fit_raises(self):
        X = np.asfortranarray(np.array([[0], [1]], dtype=np.uint8))
        model = SetCoveringMachineClassifier()
        with self.assertRaisesRegex(RuntimeError, "must be fitted"):
            model.predict(X)

    def test_fit_rejects_non_uint8_inputs(self):
        model = SetCoveringMachineClassifier(max_rules=2)
        X = np.array([[0.0], [1.0]], dtype=np.float64)
        y = np.array([0, 1], dtype=np.uint8)
        with self.assertRaisesRegex(TypeError, r"X must have dtype np.uint8"):
            model.fit(X, y)

    def test_fit_rejects_non_f_contiguous_inputs(self):
        model = SetCoveringMachineClassifier(max_rules=2)
        X_base = np.array([[0, 9], [1, 9], [2, 9], [3, 9]], dtype=np.uint8)
        X = X_base[:, :1]
        self.assertFalse(X.flags.f_contiguous)
        y = np.array([0, 0, 1, 1], dtype=np.uint8)
        with self.assertRaisesRegex(ValueError, r"X must be F-contiguous"):
            model.fit(X, y)

    def test_fit_rejects_non_uint8_y(self):
        model = SetCoveringMachineClassifier(max_rules=2)
        X = np.asfortranarray(np.array([[0], [1]], dtype=np.uint8))
        y = np.array([0, 1], dtype=np.int64)
        with self.assertRaisesRegex(TypeError, r"y must have dtype np.uint8"):
            model.fit(X, y)

    def test_fit_rejects_non_c_contiguous_y(self):
        model = SetCoveringMachineClassifier(max_rules=2)
        X = np.asfortranarray(np.array([[0], [1]], dtype=np.uint8))
        y = np.array([0, 1, 1, 0], dtype=np.uint8)[::2]
        self.assertFalse(y.flags.c_contiguous)
        with self.assertRaisesRegex(ValueError, r"y must be C-contiguous"):
            model.fit(X, y)

    def test_fit_rejects_non_binary_y(self):
        model = SetCoveringMachineClassifier(max_rules=2)
        X = np.asfortranarray(np.array([[0], [1], [2]], dtype=np.uint8))
        y = np.array([0, 2, 2], dtype=np.uint8)
        with self.assertRaisesRegex(ValueError, r"binary labels"):
            model.fit(X, y)

    def test_decision_stump_classify_feature_values_matches_classify(self):
        X = np.asfortranarray(np.array([[0, 2], [1, 1], [2, 0]], dtype=np.uint8))
        stump = DecisionStump(feature_idx=1, threshold=1, kind="greater")

        np.testing.assert_array_equal(
            stump.classify_feature_values(X[:, 1]),
            stump.classify(X),
        )

    def test_no_sklearn_or_six_imported_by_package(self):
        sys.modules.pop("pyscm", None)
        importlib.import_module("pyscm")

        loaded = set(sys.modules)
        self.assertFalse(any(name.startswith("sklearn") for name in loaded))
        self.assertFalse(any(name == "six" or name.startswith("six.") for name in loaded))
