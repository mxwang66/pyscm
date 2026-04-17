from __future__ import print_function, division, absolute_import, unicode_literals

import numpy as np

from unittest import TestCase

from pyscm._scm_utility import find_max
from pyscm.scm import SetCoveringMachineClassifier


class _TrackingSCM(SetCoveringMachineClassifier):
    def __init__(self, *args, **kwargs):
        super(_TrackingSCM, self).__init__(*args, **kwargs)
        self._fit_call_ids = []

    def _get_best_utility_rules(self, Xt, y, X_argsort_by_feature_T, example_idx):
        self._fit_call_ids.append((id(Xt), id(y), id(X_argsort_by_feature_T), id(example_idx)))
        return super(_TrackingSCM, self)._get_best_utility_rules(Xt, y, X_argsort_by_feature_T, example_idx)


class UtilityTests(TestCase):
    def setUp(self):
        """
        Called before each test

        """
        pass

    def tearDown(self):
        """
        Called after each test

        """
        pass

    def test_1(self):
        """
        Dummy test #1
        """
        X = np.asfortranarray(np.array([[1, 2, 2, 2, 3, 4]], dtype=np.uint8).reshape(-1, 1).copy())
        y = np.array([0, 1, 0, 1, 1, 1], dtype=np.uint8)
        p = 1
        Xas = np.ascontiguousarray(np.argsort(X, axis=0).T, dtype=np.intp)
        (
            best_utility,
            best_feat_idx,
            best_thresholds,
            best_kinds,
            best_N,
            best_P_bar,
        ) = find_max(p, X.T, y, Xas, np.arange(X.shape[0], dtype=np.intp))
        np.testing.assert_almost_equal(actual=best_utility, desired=1.0)
        np.testing.assert_almost_equal(actual=best_feat_idx, desired=[0])
        np.testing.assert_almost_equal(actual=best_thresholds, desired=[1])
        np.testing.assert_almost_equal(actual=best_kinds, desired=[0])

    def test_2(self):
        """
        Test that hyperparameter p works
        """
        X = np.asfortranarray(np.array([[1, 2, 2, 2, 3, 4]], dtype=np.uint8).reshape(-1, 1).copy())
        y = np.array([0, 1, 0, 1, 1, 1], dtype=np.uint8)
        Xas = np.ascontiguousarray(np.argsort(X, axis=0).T, dtype=np.intp)
        p = 0.5
        (
            best_utility,
            best_feat_idx,
            best_thresholds,
            best_kinds,
            best_N,
            best_P_bar,
        ) = find_max(p, X.T, y, Xas, np.arange(X.shape[0], dtype=np.intp))

        np.testing.assert_almost_equal(actual=best_utility, desired=1.0)
        np.testing.assert_almost_equal(actual=best_feat_idx, desired=[0, 0])
        np.testing.assert_almost_equal(actual=best_thresholds, desired=[1, 2])
        np.testing.assert_almost_equal(actual=best_kinds, desired=[0, 0])

    def test_3(self):
        """
        Test that example_idx works
        """
        X = np.asfortranarray(np.array([[1, 1], [0, 0], [1, 0]], dtype=np.uint8))
        y = np.array([0, 1, 1], dtype=np.uint8)
        Xas = np.ascontiguousarray(np.argsort(X, axis=0).T, dtype=np.intp)
        p = 1.0

        # If example 3 is included, the best feature is feat1
        (
            best_utility,
            best_feat_idx,
            best_thresholds,
            best_kinds,
            best_N,
            best_P_bar,
        ) = find_max(p, X.T, y, Xas, np.arange(X.shape[0], dtype=np.intp))
        np.testing.assert_almost_equal(actual=best_feat_idx, desired=[1])

        # If example 3 is included, the best feature is feat1
        (
            best_utility,
            best_feat_idx,
            best_thresholds,
            best_kinds,
            best_N,
            best_P_bar,
        ) = find_max(p, X.T, y, Xas, np.array([1, 2], dtype=np.intp))
        np.testing.assert_almost_equal(actual=best_feat_idx, desired=[0, 1])

    def test_4(self):
        """
        Test that solver return accurate equivalent rules
        """
        X = np.asfortranarray(np.array(
            [
                [1, 1, 5, 1],
                [2, 1, 5, 1],
                [2, 1, 5, 1],
                [3, 1, 17, 0],
                [4, 1, 17, 0],
                [5, 1, 17, 0],
                [6, 1, 17, 0],
                [7, 1, 17, 0],
            ],
            dtype=np.uint8,
        ))
        y = np.array([0, 0, 0, 1, 1, 1, 1, 1], dtype=np.uint8)
        Xas = np.ascontiguousarray(np.argsort(X, axis=0).T, dtype=np.intp)
        p = 1.0

        (
            best_utility,
            best_feat_idx,
            best_thresholds,
            best_kinds,
            best_N,
            best_P_bar,
        ) = find_max(p, X.T, y, Xas, np.arange(X.shape[0], dtype=np.intp))
        np.testing.assert_almost_equal(actual=best_utility, desired=3.0)
        np.testing.assert_almost_equal(actual=best_feat_idx, desired=[0, 2, 3])
        np.testing.assert_almost_equal(actual=best_thresholds, desired=[2, 5, 0])
        np.testing.assert_almost_equal(actual=best_kinds, desired=[0, 0, 1])

    def test_5(self):
        """
        Test that solver return accurate N and P_bar
        """
        X = np.asfortranarray(np.array(
            [
                [0, 5, 0],
                [0, 17, 0],
                [1, 5, 0],
                [0, 17, 0],
                [1, 5, 0],
                [1, 17, 0],
                [1, 17, 0],
                [1, 17, 0],
            ],
            dtype=np.uint8,
        ))
        y = np.array([0, 0, 0, 1, 1, 1, 1, 1], dtype=np.uint8)
        Xas = np.ascontiguousarray(np.argsort(X, axis=0).T, dtype=np.intp)
        p = 1.0

        (
            best_utility,
            best_feat_idx,
            best_thresholds,
            best_kinds,
            best_N,
            best_P_bar,
        ) = find_max(p, X.T, y, Xas, np.arange(X.shape[0], dtype=np.intp))
        np.testing.assert_almost_equal(actual=best_N, desired=[2, 2])
        np.testing.assert_almost_equal(actual=best_P_bar, desired=[1, 1])


    def test_direct_call_find_max_signature(self):
        """Direct extension call should support the strict uint8/int64 signature."""
        X = np.asfortranarray(np.array([[1], [2], [3]], dtype=np.uint8))
        y = np.array([0, 1, 1], dtype=np.uint8)
        Xas = np.ascontiguousarray(np.argsort(X, axis=0).T, dtype=np.intp)

        result = find_max(1.0, X.T, y, Xas, np.arange(X.shape[0], dtype=np.intp))

        self.assertEqual(len(result), 6)
        np.testing.assert_array_equal(result[1], np.array([0], dtype=np.int64))

    def test_direct_call_find_max_rejects_non_transposed_layout(self):
        """Direct extension call should reject non-C-contiguous Xt inputs."""
        X = np.asfortranarray(np.array([[1], [2], [3]], dtype=np.uint8))
        y = np.array([0, 1, 1], dtype=np.uint8)
        Xas = np.ascontiguousarray(np.argsort(X, axis=0).T, dtype=np.intp)

        with self.assertRaises(TypeError):
            find_max(1.0, X, y, Xas, np.arange(X.shape[0], dtype=np.intp))

    def test_estimator_fit_still_works(self):
        """High-level estimator fit path should still work."""
        X = np.asfortranarray(np.array([[0], [1], [2], [3]], dtype=np.uint8))
        y = np.array([0, 0, 1, 1], dtype=np.uint8)

        model = SetCoveringMachineClassifier(max_rules=2, random_state=0)
        model.fit(X, y)

        self.assertGreaterEqual(len(model.model_), 1)

    def test_estimator_ignores_utility_feature_weights_fit_param(self):
        """Estimator behavior with utility__* fit params remains unchanged."""
        X = np.asfortranarray(np.array([[0], [1], [2], [3]], dtype=np.uint8))
        y = np.array([0, 0, 1, 1], dtype=np.uint8)
        model = SetCoveringMachineClassifier(max_rules=2, random_state=0)
        model.fit(X, y, utility__feature_weights=np.array([1.0], dtype=np.double))
        self.assertGreaterEqual(len(model.model_), 1)


    def test_fit_reuses_main_training_buffers(self):
        """fit should avoid per-iteration copies of full training arrays."""
        X = np.asfortranarray(np.array(
            [
                [0, 0],
                [0, 1],
                [1, 0],
                [1, 1],
                [2, 0],
                [2, 1],
            ],
            dtype=np.uint8,
        ))
        y = np.array([0, 0, 1, 1, 0, 1], dtype=np.uint8)

        model = _TrackingSCM(max_rules=3, random_state=0)
        model.fit(X, y)

        self.assertGreaterEqual(len(model._fit_call_ids), 1)
        self.assertEqual(len({call[0] for call in model._fit_call_ids}), 1)
        self.assertEqual(len({call[1] for call in model._fit_call_ids}), 1)
        self.assertEqual(len({call[2] for call in model._fit_call_ids}), 1)

    def test_random_data(self):
        """
        Random testing
        """
        n_tests = 10  # 10000

        # The more examples, the more likely we are to have equal feature values
        for n_examples in [10, 100, 1000]:

            # Do this a few times for each configuration
            for _ in range(n_tests):
                p = max(0, np.random.rand() * 100.0)
                x = np.asfortranarray(np.random.randint(0, 6, size=(n_examples, 1), dtype=np.uint8))
                xas = np.ascontiguousarray(np.argsort(x, axis=0).T, dtype=np.intp)
                y = np.random.randint(0, 2, n_examples, dtype=np.uint8)
                thresholds = np.unique(x)

                # Use the solver to find the solution
                (
                    solver_best_utility,
                    solver_best_feat_idx,
                    solver_best_thresholds,
                    solver_best_kinds,
                    solver_best_N,
                    solver_best_P_bar,
                ) = find_max(p, x.T, y, xas, np.arange(n_examples, dtype=np.intp))

                # Less equal rule utilities
                le_rule_utilities = []
                for t in thresholds:
                    rule_classifications = (x <= t).reshape(
                        -1,
                    )
                    N = (~rule_classifications[y == 0]).sum()
                    P_bar = (~rule_classifications[y == 1]).sum()
                    le_rule_utilities.append(N - p * P_bar)

                # Greater rule utilities
                g_rule_utilities = []
                for t in thresholds:
                    rule_classifications = (x > t).reshape(
                        -1,
                    )
                    N = (~rule_classifications[y == 0]).sum()
                    P_bar = (~rule_classifications[y == 1]).sum()
                    g_rule_utilities.append(N - p * P_bar)

                np.testing.assert_almost_equal(
                    actual=solver_best_utility,
                    desired=max(max(le_rule_utilities), max(g_rule_utilities)),
                )

    def test_fit_rejects_non_uint8_inputs(self):
        model = SetCoveringMachineClassifier(max_rules=2, random_state=0)
        X = np.array([[0.0], [1.0]], dtype=np.float64)
        y = np.array([0, 1], dtype=np.uint8)
        with self.assertRaisesRegex(TypeError, r"X must have dtype np.uint8"):
            model.fit(X, y)

    def test_fit_rejects_non_f_contiguous_inputs(self):
        model = SetCoveringMachineClassifier(max_rules=2, random_state=0)
        X_base = np.array([[0, 9], [1, 9], [2, 9], [3, 9]], dtype=np.uint8)
        X = X_base[:, :1]
        self.assertFalse(X.flags.f_contiguous)
        y = np.array([0, 0, 1, 1], dtype=np.uint8)
        with self.assertRaisesRegex(ValueError, r"X must be F-contiguous"):
            model.fit(X, y)
