#include <cstdint>
#include <stdexcept>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "best_utility.h"
#include "solver.h"

namespace py = pybind11;

template <typename T>
py::array_t<T, py::array::c_style> ensure_c_style_array(
        const py::handle &obj,
        const char *name) {
    auto fast_path = py::array_t<T, py::array::c_style>::ensure(obj);
    if (fast_path) {
        return fast_path;
    }

    try {
        return py::cast<py::array_t<T, py::array::c_style | py::array::forcecast>>(obj);
    } catch (const py::cast_error &) {
        throw py::type_error(std::string(name) + " has an incompatible dtype");
    }
}

py::tuple find_max_binding(
        double p,
        py::object X_obj,
        py::object y_obj,
        py::object X_argsort_by_feature_T_obj,
        py::object example_idx_obj,
        py::object feature_weights_obj) {
    auto X = ensure_c_style_array<double>(X_obj, "X");
    auto y = ensure_c_style_array<std::int64_t>(y_obj, "y");
    auto X_argsort_by_feature_T = ensure_c_style_array<std::int64_t>(X_argsort_by_feature_T_obj, "X_argsort_by_feature_T");
    auto example_idx = ensure_c_style_array<std::int64_t>(example_idx_obj, "example_idx");

    if (X.ndim() != 2) {
        throw py::type_error("X must be a 2D numpy.ndarray");
    }
    if (y.ndim() != 1) {
        throw py::type_error("y must be a 1D numpy.ndarray");
    }
    if (X_argsort_by_feature_T.ndim() != 2) {
        throw py::type_error("X_argsort_by_feature_T must be a 2D numpy.ndarray");
    }
    if (example_idx.ndim() != 1) {
        throw py::type_error("example_idx must be a 1D numpy.ndarray");
    }

    const auto n_examples = static_cast<std::int64_t>(X.shape(0));
    const auto n_features = static_cast<std::int64_t>(X.shape(1));

    if (static_cast<std::int64_t>(y.shape(0)) != n_examples) {
        throw py::type_error("X and y must have the same number of rows");
    }
    if (static_cast<std::int64_t>(X_argsort_by_feature_T.shape(0)) != n_features) {
        throw py::type_error("X must have as many columns as X_argsort_by_feature_T has rows");
    }
    if (static_cast<std::int64_t>(X_argsort_by_feature_T.shape(1)) != n_examples) {
        throw py::type_error("X must have as many rows as X_argsort_by_feature_T has columns");
    }

    py::array_t<double, py::array::c_style> feature_weights_array;
    std::vector<double> default_feature_weights;

    const double *feature_weights_data = nullptr;
    if (feature_weights_obj.is_none()) {
        default_feature_weights.assign(static_cast<size_t>(n_features), 1.0);
        feature_weights_data = default_feature_weights.data();
    } else {
        feature_weights_array = ensure_c_style_array<double>(feature_weights_obj, "feature_weights");
        if (feature_weights_array.ndim() != 1) {
            throw py::type_error("feature_weights must be a 1D numpy.ndarray");
        }
        if (static_cast<std::int64_t>(feature_weights_array.shape(0)) != n_features) {
            throw py::type_error("feature_weights must have shape X.shape[1]");
        }
        feature_weights_data = feature_weights_array.data();
    }

    BestUtility best_solution(100);
    const int status = find_max(
            p,
            X.data(),
            y.data(),
            X_argsort_by_feature_T.data(),
            example_idx.data(),
            feature_weights_data,
            static_cast<std::int64_t>(example_idx.shape(0)),
            n_examples,
            n_features,
            best_solution);

    if (status != 0) {
        throw std::runtime_error("An error occurred in the solver");
    }

    const py::ssize_t n_equiv = best_solution.best_n_equiv;
    py::array_t<std::int64_t> opti_feat_idx(n_equiv);
    py::array_t<double> opti_thresholds(n_equiv);
    py::array_t<std::int64_t> opti_kinds(n_equiv);
    py::array_t<std::int64_t> opti_N(n_equiv);
    py::array_t<std::int64_t> opti_P_bar(n_equiv);

    auto opti_feat_idx_mut = opti_feat_idx.mutable_unchecked<1>();
    auto opti_thresholds_mut = opti_thresholds.mutable_unchecked<1>();
    auto opti_kinds_mut = opti_kinds.mutable_unchecked<1>();
    auto opti_N_mut = opti_N.mutable_unchecked<1>();
    auto opti_P_bar_mut = opti_P_bar.mutable_unchecked<1>();

    for (py::ssize_t i = 0; i < n_equiv; ++i) {
        opti_feat_idx_mut(i) = best_solution.best_feat_idx[i];
        opti_thresholds_mut(i) = best_solution.best_feat_threshold[i];
        opti_kinds_mut(i) = best_solution.best_feat_kind[i];
        opti_N_mut(i) = best_solution.best_N[i];
        opti_P_bar_mut(i) = best_solution.best_P_bar[i];
    }

    return py::make_tuple(
            best_solution.best_utility,
            opti_feat_idx,
            opti_thresholds,
            opti_kinds,
            opti_N,
            opti_P_bar);
}

PYBIND11_MODULE(_scm_utility, m) {
    m.doc() = "SCM utility maximization solver";
    m.def(
            "find_max",
            &find_max_binding,
            py::arg("p"),
            py::arg("X"),
            py::arg("y"),
            py::arg("X_argsort_by_feature_T"),
            py::arg("example_idx"),
            py::arg("feature_weights") = py::none(),
            "Find the split of maximum utility.");
}
