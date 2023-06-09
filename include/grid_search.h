#include <array>
#include <type_traits>
#include <iostream>
#pragma once

namespace gridsearch {

    struct Hyperparameters {
        float learning_rate;
        int batch_size;
        int num_hidden_units;
        int num_epochs;
    };

    template <auto StartNumerator, auto StartDenominator, auto IncrementNumerator, auto IncrementDenominator, std::size_t... Is>
    constexpr auto generate_sequence_impl_fractional(std::index_sequence<Is...>) {
        return std::array<float, sizeof...(Is)>{{( ((float)StartNumerator/(float)StartDenominator) + ((float)IncrementNumerator/(float)IncrementDenominator) * Is)...}};
    }

    template <int Start, int Increment, std::size_t... Is>
    constexpr auto generate_sequence_impl(std::index_sequence<Is...>) {
        return std::array<int, sizeof...(Is)>{{( Start + Increment * Is)...}};
    }
    template <float Start, float Increment, std::size_t... Is>
    constexpr auto generate_sequence_impl_float(std::index_sequence<Is...>) {
        return std::array<float, sizeof...(Is)>{{( Start + Increment * Is)...}};
    }
    template <float Start, float Increment, std::size_t N>
    constexpr auto generate_sequence_float() {
        return generate_sequence_impl_float<Start,Increment>(std::make_index_sequence<N>{});
    }


    template <auto StartNumerator, auto StartDenominator, auto IncrementNumerator, auto IncrementDenominator, std::size_t N>
    constexpr auto generate_sequence_fractional() {
        return generate_sequence_impl_fractional<StartNumerator, StartDenominator, IncrementNumerator, IncrementDenominator>(std::make_index_sequence<N>{});
    }

    template <auto Start, auto Increment, std::size_t N>
    constexpr auto generate_sequence() {
        return generate_sequence_impl<Start,Increment>(std::make_index_sequence<N>{});
    }


    template<std::size_t N1, std::size_t N2, std::size_t N3, std::size_t N4>
    constexpr auto generateHyperparameters(std::array<float, N1> learning_rates, std::array<int, N2> batch_sizes, std::array<int, N3> num_hidden_units_array, std::array<int, N4> num_epochs_array) {
        std::array<Hyperparameters, N1*N2*N3*N4> product = {};

        std::size_t index = 0;
        for(const auto &lr : learning_rates) {
            for(const auto &bs : batch_sizes) {
                for(const auto &nhu : num_hidden_units_array) {
                    for(const auto &ne : num_epochs_array) {
                        product[index++] = {lr, bs, nhu, ne};
                    }
                }
            }
        }

        return product;
    }

    constexpr auto learning_rates = generate_sequence_fractional<25, 100000, -1, 100000, 25>();
    // constexpr auto learning_rates = generate_sequence_float<2.5e-5f, -0.1e-5f, 25>();

    constexpr auto batch_sizes = std::array{0};
    constexpr auto num_hidden_units_array = std::array{0};
    constexpr auto num_epochs_array = generate_sequence<20, 5, 8>();

    constexpr auto hyperparameters = generateHyperparameters(
        learning_rates,
        batch_sizes,
        num_hidden_units_array,
        num_epochs_array
    );

}
