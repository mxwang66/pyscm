#include <cstdint>
#include <stdexcept>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "best_utility.h"
#include "solver.h"

namespace py = pybind11;

py::tuple find_max_binding(
        double p,
        py::object X_obj,
        py::object y_obj,
        py::object X_argsort_by_feature_T_obj,
        py::object example_idx_obj) {
    auto X = py::reinterpret_borrow<py::array>(X_obj);
    auto y = py::reinterpret_borrow<py::array>(y_obj);
    auto X_argsort_by_feature_T = py::reinterpret_borrow<py::array>(X_argsort_by_feature_T_obj);
    auto example_idx = py::reinterpret_borrow<py::array>(example_idx_obj);

    if (!X.dtype().is(py::dtype::of<std::uint8_t>())) {
        throw py::type_error("X has an incompatible dtype");
    }
    if (!y.dtype().is(py::dtype::of<std::uint8_t>())) {
        throw py::type_error("y has an incompatible dtype");
    }
    if (!X_argsort_by_feature_T.dtype().is(py::dtype::of<std::int64_t>())) {
        throw py::type_error("X_argsort_by_feature_T has an incompatible dtype");
    }
    if (!example_idx.dtype().is(py::dtype::of<std::int64_t>())) {
        throw py::type_error("example_idx has an incompatible dtype");
    }
    if (!(X.flags() & py::array::c_style)) {
        throw py::type_error("X must be C-contiguous");
    }
    if (!(y.flags() & py::array::c_style)) {
        throw py::type_error("y must be C-contiguous");
    }
    if (!(X_argsort_by_feature_T.flags() & py::array::c_style)) {
        throw py::type_error("X_argsort_by_feature_T must be C-contiguous");
    }
    if (!(example_idx.flags() & py::array::c_style)) {
        throw py::type_error("example_idx must be C-contiguous");
    }

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

    const auto *X_data = static_cast<const std::uint8_t *>(X.data());
    const auto *y_data = static_cast<const std::uint8_t *>(y.data());
    const auto *Xas_data = static_cast<const std::int64_t *>(X_argsort_by_feature_T.data());
    const auto *example_idx_data = static_cast<const std::int64_t *>(example_idx.data());

    BestUtility best_solution(100);
    const int status = find_max(
            p,
            X_data,
            y_data,
            Xas_data,
            example_idx_data,
            static_cast<std::int64_t>(example_idx.shape(0)),
            n_examples,
            n_features,
            best_solution);

    if (status != 0) {
        throw std::runtime_error("An error occurred in the solver");
    }

    const py::ssize_t n_equiv = best_solution.best_n_equiv;
    py::array_t<std::int64_t> opti_feat_idx(n_equiv);
    py::array_t<std::uint8_t> opti_thresholds(n_equiv);
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
            "Find the split of maximum utility.");
}
