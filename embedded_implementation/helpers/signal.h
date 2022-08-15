#ifndef SIGNAL_H
#define SIGNAL_H

#include "globals.h"
#include <iostream>
#include <iterator>
#include <numbers>
#include <array>
#include <random>
#include <chrono>
#include <execution>
#include <type_traits>
#include <vector>
#include "print_helpers.h"


/*
 * constexpr to create a filter bank for MFCC of size N/2 and MEL_N
 * We will create 26 filters but in the future use only 20
 */
constexpr std::array<real_fft_array_t, MEL_N> mfcc_filters() {
    // Création de 24 filtres se recoupant allant de 20Hz à 22050Hz
    // Soit 31mels à 3923 mels d'aprés l'équations m=2595*log10(1+f/700)
    // On obtient:
    const int mel_min = 31;
    const int mel_max = 3923;

    double mel_inc = (mel_max - mel_min) / (real_t) (MEL_N + 1);
    std::size_t index, sous_index;
    std::array<real_t, MEL_N> mel_centers;
    std::array<real_t, MEL_N> fcenters_norm;
    std::array<real_fft_array_t, MEL_N> w = {}; // ToReturn
    double index_start[MEL_N];   //FFT index for the first sample of each filter
    double index_stop[MEL_N];    //FFT index for the last sample of each filter
    float increment, decrement; // increment and decrement of the left and right ramp
    double sum = 0.0;
    //for(index=0; index<MEL_FILTERS_N; index++){
    //    for(sous_index=0; sous_index<= FFT_SIZE/2 + 1; sous_index++){
    //        w[index][sous_index] = 0.0;
    //    }
    //}
    for (index = 1; index <= MEL_N; index++) {
        mel_centers[index - 1] = (real_t) index * mel_inc + mel_min;
        fcenters_norm[index - 1] = 700.0 * (pow(10.0, mel_centers[index - 1] / 2595.0) - 1.0);
        fcenters_norm[index - 1] = round(fcenters_norm[index - 1] / (Fs / N));
    }
    //std::cout << std::endl;
    for (index = 1; index <= (MEL_N - 1); index++) {
        index_start[index] = fcenters_norm[index - 1];
        index_stop[index - 1] = fcenters_norm[index];
    }
    index_start[0] = round((real_t) N * 20.0 / (real_t) Fs);
    index_stop[MEL_N - 1] = round((real_t) N * 22050.0 / (real_t) Fs);
    for (index = 1; index < MEL_N; index++) {
        increment = 1. / ((real_t) fcenters_norm[index - 1] - (real_t) index_start[index - 1]); //Parite montante du triangle
        for (sous_index = index_start[index - 1]; sous_index <= fcenters_norm[index - 1]; sous_index++) {
            w[index - 1][sous_index] = ((real_t) sous_index - (real_t) index_start[index - 1]) * increment;
        }
        decrement = 1. / ((real_t) index_stop[index - 1] - (real_t) fcenters_norm[index - 1]); //Partie descendante du triangle
        for (sous_index = fcenters_norm[index - 1]; sous_index <= index_stop[index - 1]; sous_index++) {
            w[index - 1][sous_index] = ((real_t) sous_index - (real_t) fcenters_norm[index - 1]) * decrement;
        }
    }
    for (index = 1; index <= MEL_N; index++) {
        for (sous_index = 1; sous_index <= FFT_SIZE; sous_index++) {
            sum = sum + w[index - 1][sous_index - 1];
        }
        for (sous_index = 1; sous_index <= FFT_SIZE; sous_index++) {
            w[index - 1][sous_index - 1] = w[index - 1][sous_index - 1] / sum;
        }
        sum = 0.0;

    }
    return w;
}


/*
 * constexpr to create a hamming window of size N
 */
constexpr real_n_array_t hamming_window() {
    real_n_array_t w;
    std::generate(w.begin(), w.end(),
                  [&, index = -1]()mutable {
                      index++;
                      return (0.54 - 0.46 * std::cos(2 * std::numbers::pi * index / (N - 1)));
                  });
    return w;
}

/*
 * apply the given window to the given array
 * real_n_array_t w -> window
 * complex_n_array_t a -> array to apply window
 */
static inline void windowing(const real_n_array_t &w, complex_n_array_t &a) {
    std::transform(a.cbegin(),
                   a.cend(),
                   a.begin(),
                   [&, index = -1](complex_t c)mutable {
                       index++;
                       return w[index] * c;
                   });
}

/*
 * Merged data after Fourier Tranform with a filter bank created by the constexpr mfcc_filters
 * The output is a real_vector_t (std::vector<real_t>) with a length equal to the number of filters
 * We take the log of all values in order to perform a non linear rectification
 */
real_vector_t apply_filterbank(std::array<real_fft_array_t, MEL_APPLIED_N> filterbank, real_fft_array_t e) {
    real_vector_t value_filtering;
    double filter_value;
    for (real_fft_array_t filter: filterbank) {
        filter_value = std::transform_reduce(e.cbegin(),
                                             e.cend(),
                                             filter.cbegin(),
                                             0.0,
                                             std::plus<>(),
                                             std::multiplies<>());
        value_filtering.push_back(std::log(std::max(filter_value, 2e-22)));
    }
    return value_filtering;
}

/*
 * Discrete Cosine Transform
 * Input data: vector after MFCC filter
 * It expresses a finite sequence of data points in terms of a sum of cosine functions
 * oscillating at different frequencies.
 */
std::array<real_t, MEL_APPLIED_N> dct(real_vector_t e) {
    std::array<real_t, MEL_APPLIED_N> cepstrum{};
    size_t sous_index, index;
    for (index = 1; index <= MEL_APPLIED_N; index++) {
        cepstrum[index - 1] = 0.0;
        for (sous_index = 1; sous_index <= MEL_APPLIED_N; sous_index++) {
            cepstrum[index - 1] = cepstrum[index - 1] + e[sous_index - 1] *
                                                        cos(M_PI * ((real_t) index) / ((real_t) MEL_APPLIED_N) *
                                                            ((real_t) sous_index - 0.5));
            cepstrum[index - 1] = sqrt(2.0 / (real_t) MEL_APPLIED_N) * cepstrum[index - 1];
        }
    }
    return cepstrum;
}

/*
 * using DCT-2 formula from https://en.wikipedia.org/wiki/Discrete_cosine_transform
 * Based on p9-p10 https://eeweb.engineering.nyu.edu/~yao/EE3414/ImageCoding_DCT.pdf where formula inside cos() is equal to wiki dtc-2 inside cos
 */
real_vector_t dct2(const real_vector_t &v_in) {
    real_vector_t v_out = {};
    for (std::size_t k = 0; k < v_in.size(); k++) {
        real_vector_t u_k = {};
        real_t a = k > 0 ? std::sqrt(2.0 / (real_t) v_in.size()) : std::sqrt(1.0 / (real_t) v_in.size());
        for (std::size_t n = 0; n < v_in.size(); n++) {
            u_k.push_back(a * std::cos(M_PI / (real_t) v_in.size() * ((real_t) n + 0.5) * (real_t) k));
        }

        real_t t_k = std::transform_reduce(v_in.cbegin(), v_in.cend(), u_k.cbegin(), 0.0, std::plus<>(), std::multiplies<>());
        v_out.push_back(t_k);
    }

    return v_out;
}


/*
 * the root-of-unity complex multiplicative constants in the butterfly operations of the  FFT algorithm,
 * used to recursively combine smaller discrete Fourier transforms.
 * This remains the term's most common meaning, but it may also be used for any data-independent
 * multiplicative constant in an FFT.
 */
constexpr complex_fft_array_t twiddle_factors() {
    std::array<complex_t, N / 2> t;
    for (std::size_t k = 0; k < N / 2; k++) {
        t[k] = std::polar(1.0, -2.0 * std::numbers::pi * k / N);
    }
    return t;
}

/*
 * In order to perform FFT we changed the bit order
 * Use the butterfly algorithm
 */
constexpr std::array<std::size_t, N> bit_reverse_array() {
    std::array<std::size_t, N> unscrambled{};
    std::size_t m = std::log2(N);
    //std::size_t m = ((unsigned) (8 * sizeof(unsigned long long) - __builtin_clzll((N)) - 1));
    for (std::size_t i = 0; i < N; i++) {
        std::size_t j = i;
        j = (((j & 0xaaaaaaaa) >> 1) | ((j & 0x55555555) << 1));
        j = (((j & 0xcccccccc) >> 2) | ((j & 0x33333333) << 2));
        j = (((j & 0xf0f0f0f0) >> 4) | ((j & 0x0f0f0f0f) << 4));
        j = (((j & 0xff00ff00) >> 8) | ((j & 0x00ff00ff) << 8));
        j = ((j >> 16) | (j << 16)) >> (32 - m);
        if (i < j) {
            unscrambled[i] = j;
        } else
            unscrambled[i] = i;
    }
    return unscrambled;
}

/*
 * apply ite dit fft to given array
 * complex_n_array_t &x : given complex array
 */
static inline void ite_dit_fft(complex_n_array_t &x) {
    std::size_t problemSize = x.size();
    std::size_t stages = std::log2(problemSize);
    auto tf = twiddle_factors();

    constexpr std::array<std::size_t, N> unscrambled = bit_reverse_array();
    for (std::size_t i = 0; i < x.size(); i++) {
        std::size_t j = unscrambled[i];
        if (i < j) {
            swap(x[i], x[j]);
        }
    }

    for (std::size_t stage = 0; stage <= stages; stage++) {
        std::size_t currentSize = 1 << stage;
        std::size_t step = stages - stage;
        std::size_t halfSize = currentSize / 2;
        for (std::size_t k = 0; k < problemSize; k = k + currentSize) {
            //for (std::size_t k = 0; k <= problemSize - currentSize; k = k + currentSize) {
            for (std::size_t j = 0; j < halfSize; j++) {
                auto u = x[k + j];
                auto v = x[k + j + halfSize] * tf[j * (1 << step)];
                x[k + j] = (u + v);
                x[k + j + halfSize] = (u - v);
            }
        }
    }
}

#endif //SIGNAL_H
