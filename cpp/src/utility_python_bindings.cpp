#include <cstdint>
#include <stdexcept>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "best_utility.h"
#include "solver.h"

namespace py = pybind11;

py::tuple find_max_binding(
        double p,
        py::array_t<std::uint8_t, py::array::c_style> Xt,
        py::array_t<std::uint8_t, py::array::c_style> y,
        py::array_t<std::int64_t, py::array::c_style> example_idx) {

    if (Xt.ndim() != 2 || y.ndim() != 1 || example_idx.ndim() != 1) {
        throw py::type_error("Unexpected array dimensions passed to find_max.");
    }

    const auto n_features = static_cast<std::int64_t>(Xt.shape(0));
    const auto n_examples = static_cast<std::int64_t>(Xt.shape(1));

    if (static_cast<std::int64_t>(y.shape(0)) != n_examples) {
        throw py::type_error("Xt and y must have compatible dimensions");
    }

    const auto *Xt_data = Xt.data();
    const auto *y_data = y.data();
    const auto *example_idx_data = example_idx.data();

    BestUtility best_solution(100);
    const int status = find_max(
            p,
            Xt_data,
            y_data,
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
            py::arg("Xt"),
            py::arg("y"),
            py::arg("example_idx"),
            "Find the split of maximum utility.");
}
