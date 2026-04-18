import importlib
import sys
from unittest import TestCase
from unittest.mock import patch

import numpy as np

from pyscm._scm_utility import find_max
from pyscm.scm import DecisionStump, SetCoveringMachineClassifier


class UtilityTests(TestCase):
    @staticmethod
    def _mock_find_max_rule_greater_than_one(*_args, **_kwargs):
        return (
            1.0,
            np.array([0], dtype=np.int64),
            np.array([1], dtype=np.uint8),
            np.array([0], dtype=np.uint8),  # "greater"
            np.array([1], dtype=np.int64),
            np.array([0], dtype=np.int64),
        )

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
        y = np.array([False, False, True, True], dtype=np.bool_)

        model = SetCoveringMachineClassifier(model_type="conjunction", max_rules=2)
        model.fit(X, y)
        preds = model.predict(X)

        self.assertEqual(preds.dtype, np.uint8)
        self.assertEqual(preds.shape, y.shape)

    def test_fit_predict_disjunction(self):
        X = np.asfortranarray(np.array([[0], [1], [2], [3]], dtype=np.uint8))
        y = np.array([False, False, True, True], dtype=np.bool_)

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

    def test_fit_rejects_uint8_y(self):
        model = SetCoveringMachineClassifier(max_rules=2)
        X = np.asfortranarray(np.array([[0], [1]], dtype=np.uint8))
        y = np.array([0, 1], dtype=np.uint8)
        with self.assertRaisesRegex(TypeError, r"y must have dtype np.bool_"):
            model.fit(X, y)

    def test_fit_rejects_non_bool_y(self):
        model = SetCoveringMachineClassifier(max_rules=2)
        X = np.asfortranarray(np.array([[0], [1]], dtype=np.uint8))
        y = np.array([0, 1], dtype=np.int64)
        with self.assertRaisesRegex(TypeError, r"y must have dtype np.bool_"):
            model.fit(X, y)

    def test_fit_rejects_non_c_contiguous_y(self):
        model = SetCoveringMachineClassifier(max_rules=2)
        X = np.asfortranarray(np.array([[0], [1]], dtype=np.uint8))
        y = np.array([False, True, True, False], dtype=np.bool_)[::2]
        self.assertFalse(y.flags.c_contiguous)
        with self.assertRaisesRegex(ValueError, r"y must be C-contiguous"):
            model.fit(X, y)

    def test_fit_rejects_single_class_bool_y(self):
        model = SetCoveringMachineClassifier(max_rules=2)
        X = np.asfortranarray(np.array([[0], [1], [2]], dtype=np.uint8))
        y = np.array([True, True, True], dtype=np.bool_)
        with self.assertRaisesRegex(ValueError, r"both boolean labels"):
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

    @patch("pyscm.scm.find_max")
    def test_disjunction_uses_training_rule_for_filtering(self, mock_find_max):
        mock_find_max.side_effect = self._mock_find_max_rule_greater_than_one

        X = np.asfortranarray(np.array([[0], [1], [2], [3]], dtype=np.uint8))
        y = np.array([True, True, False, False], dtype=np.bool_)

        model = SetCoveringMachineClassifier(model_type="disjunction", max_rules=2)
        model.fit(X, y)

        # Correct behavior: training-space rule (> 1) clears remaining negatives in one iteration.
        self.assertEqual(mock_find_max.call_count, 1)
        self.assertEqual(len(model.model_.rules), 1)

        # Disjunction stores the inverse of the training-space rule in the final model.
        self.assertEqual(model.model_.rules[0].kind, "less_equal")
        self.assertEqual(model.model_.rules[0].feature_idx, 0)
        self.assertEqual(model.model_.rules[0].threshold, 1)

    @patch("pyscm.scm.find_max")
    def test_disjunction_max_rules_one_and_conjunction_rule_storage(self, mock_find_max):
        mock_find_max.side_effect = self._mock_find_max_rule_greater_than_one
        X = np.asfortranarray(np.array([[0], [1], [2], [3]], dtype=np.uint8))
        y = np.array([True, True, False, False], dtype=np.bool_)

        disjunction = SetCoveringMachineClassifier(model_type="disjunction", max_rules=1)
        disjunction.fit(X, y)
        self.assertEqual(len(disjunction.model_.rules), 1)
        self.assertEqual(disjunction.model_.rules[0].kind, "less_equal")

        conjunction = SetCoveringMachineClassifier(model_type="conjunction", max_rules=1)
        conjunction.fit(X, y)
        self.assertEqual(len(conjunction.model_.rules), 1)
        self.assertEqual(conjunction.model_.rules[0].kind, "greater")

    @patch("pyscm.scm.find_max")
    def test_find_max_receives_uint8_labels_at_extension_boundary(self, mock_find_max):
        captured = {}

        def _mock_find_max(*args):
            captured["y_dtype"] = args[2].dtype
            return self._mock_find_max_rule_greater_than_one()

        mock_find_max.side_effect = _mock_find_max
        X = np.asfortranarray(np.array([[0], [1], [2], [3]], dtype=np.uint8))
        y = np.array([False, False, True, True], dtype=np.bool_)

        model = SetCoveringMachineClassifier(model_type="conjunction", max_rules=1)
        model.fit(X, y)

        self.assertEqual(captured["y_dtype"], np.uint8)
