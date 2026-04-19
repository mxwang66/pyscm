#include <array>
#include <cstdint>

#include "best_utility.h"
#include "solver.h"


/*
 * Solver
 */

void get_n_examples_by_class(const std::int64_t *example_idx, const uint8_t *y, const std::int64_t &n_examples_included,
                             std::int64_t &n_negative, std::int64_t &n_positive){
    for(std::int64_t i = 0; i < n_examples_included; i++){
        const std::int64_t idx = example_idx[i];
        if(y[idx] == 0){
            n_negative ++;
        }
        else{
            n_positive ++;
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
             const std::int64_t *example_idx,
             std::int64_t n_examples_included,
             std::int64_t n_examples,
             std::int64_t n_features,
             BestUtility &out_best_solution){

    // Find the number of positive and negative examples
    std::int64_t n_negative = 0, n_positive = 0;
    get_n_examples_by_class(example_idx, y, n_examples_included, n_negative, n_positive);

    // Utility calculations start
    for(std::int64_t i = 0; i < n_features; i++){
        std::array<std::int64_t, 256> negatives_by_value{};
        std::array<std::int64_t, 256> positives_by_value{};

        for(std::int64_t j = 0; j < n_examples_included; j++){
            const std::int64_t idx = example_idx[j];
            const uint8_t value = Xt[i * n_examples + idx];
            const std::int64_t label = y[idx];
            positives_by_value[value] += label;
            negatives_by_value[value] += 1 - label;
        }

        std::int64_t cum_negatives = 0;
        std::int64_t cum_positives = 0;
        for(std::int64_t value = 0; value < 256; value++){
            const std::int64_t negatives_at_value = negatives_by_value[value];
            const std::int64_t positives_at_value = positives_by_value[value];
            cum_negatives += negatives_at_value;
            cum_positives += positives_at_value;

            if(negatives_at_value + positives_at_value > 0){
                update_optimal_solution(out_best_solution,
                                        i,
                                        static_cast<uint8_t>(value),
                                        cum_negatives,
                                        cum_positives,
                                        p,
                                        n_negative,
                                        n_positive);
            }
        }
    }
    return 0;
}
