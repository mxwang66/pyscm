#include <cmath>
#include <cstdint>

#include "best_utility.h"
#include "solver.h"


/*
 * Solver
 */

void get_n_examples_by_class(const bool* example_is_included, const uint8_t* y, const std::int64_t &n_examples,
                             std::int64_t &n_negative, std::int64_t &n_positive){
    for(std::int64_t i = 0; i < n_examples; i++){
        if(example_is_included[i]){
            if(y[i] == 0){
                n_negative ++;
            }
            else{
                n_positive ++;
            }
        }
    }
}

void update_optimal_solution(BestUtility &best_solution, std::int64_t const &feature_idx, uint8_t const &threshold,
                             std::int64_t const &N, std::int64_t const &P_bar, double const &p,
                             std::int64_t const &n_negative,
                             std::int64_t const &n_positive){
    // Get utility for x > t and check if optimal
    double utility_0 = static_cast<double>(N) - p * static_cast<double>(P_bar);
    if(best_solution < utility_0){
        best_solution.clear();
        best_solution.set_utility(utility_0);
        best_solution.add_equivalent(feature_idx, threshold, 0, N, P_bar);
    } else if(best_solution == utility_0){
        best_solution.add_equivalent(feature_idx, threshold, 0, N, P_bar);
    }

    // Get utility for x <= t and check if optimal
    std::int64_t N_1 = n_negative - N;
    std::int64_t P_bar_1 = n_positive - P_bar;
    double utility_1 = static_cast<double>(N_1) - p * static_cast<double>(P_bar_1);
    if(best_solution < utility_1){
        best_solution.clear();
        best_solution.set_utility(utility_1);
        best_solution.add_equivalent(feature_idx, threshold, 1, N_1, P_bar_1);
    } else if(best_solution == utility_1){
        best_solution.add_equivalent(feature_idx, threshold, 1, N_1, P_bar_1);
    }
}

int find_max(double p,
             const uint8_t *Xt,
             const uint8_t *y,
             const std::int64_t *Xas,
             const std::int64_t *example_idx,
             std::int64_t n_examples_included,
             std::int64_t n_examples,
             std::int64_t n_features,
             BestUtility &out_best_solution){

    // Make a mask that tells us which examples should be considered in the utility calculations
    bool *example_is_included = new bool[n_examples];
    std::fill_n(example_is_included, n_examples, false);

    for(std::int64_t i = 0; i < n_examples_included; i++){
        example_is_included[example_idx[i]] = true;
    }

    // Find the number of positive and negative examples
    std::int64_t n_negative = 0, n_positive = 0;
    get_n_examples_by_class(example_is_included, y, n_examples, n_negative, n_positive);

    // Utility calculations start
    for(std::int64_t i = 0; i < n_features; i++){

        // For each threshold of this feature (a threshold is an example's feature value)
        std::int64_t N = 0, P_bar = 0, prev_N = 0, prev_P_bar = 0;
        uint8_t prev_threshold = 0;
        bool has_prev_threshold = false;

        for(std::int64_t j = 0; j < n_examples; j++){

            // Get the index of the next example according to the sorting
            std::int64_t idx = Xas[i * n_examples + j];

            // Consider this example only if it is included in the calculations
            if(example_is_included[idx]){

                // Get the example's label and threshold
                uint8_t label = y[idx];
                uint8_t threshold = Xt[i * n_examples + idx];

                // Wait for the last example with this threshold before computing the utilities
                if(has_prev_threshold && threshold != prev_threshold){
                    update_optimal_solution(out_best_solution, i, prev_threshold, N, P_bar, p,
                                            n_negative, n_positive);
                }

                if(label == 1){
                    P_bar = prev_P_bar + 1;
                    N = prev_N;
                }
                else{
                    P_bar = prev_P_bar;
                    N = prev_N + 1;
                }

                prev_N = N;
                prev_P_bar = P_bar;
                prev_threshold = threshold;
                has_prev_threshold = true;
            }
        }
        if(has_prev_threshold){
            update_optimal_solution(out_best_solution, i, prev_threshold, N, P_bar, p,
                                    n_negative, n_positive);
        }
    }
    delete [] example_is_included;
    return 0;
}
