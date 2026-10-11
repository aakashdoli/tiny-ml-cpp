#pragma once

#include "tinyml/baseband/complex_sample.hpp"
#include "tinyml/baseband/fixed_vector.hpp"

#include <cstddef>

namespace tinyml::baseband {

/**
 * @brief Real-time Adaptive LMS (Least Mean Squares) Channel Equalizer.
 * 
 * Mathematically, the LMS adaptive filter is Stochastic Gradient Descent 
 * applied to complex radio signals in real-time.
 * 
 * In physical layer baseband:
 * - Wireless signals bounce off buildings, creating multipath delay spread 
 *   and Inter-Symbol Interference (ISI).
 * - This equalizer continuously updates a finite impulse response (FIR) 
 *   filter to invert the channel and reconstruct clean transmitted symbols.
 * - Operates with zero heap allocations on the CPU stack.
 */
template <std::size_t NumTaps = 7>
class LMSEqualizer {
    static_assert(NumTaps > 0, "NumTaps must be positive");

public:
    explicit LMSEqualizer(float learning_rate = 0.05f) noexcept
        : mu_(learning_rate) {
        reset();
    }

    void reset() noexcept {
        for (std::size_t i = 0; i < NumTaps; ++i) {
            weights_[i] = ComplexFloat{0.0f, 0.0f};
            buffer_[i] = ComplexFloat{0.0f, 0.0f};
        }
        // Initialize central tap to 1.0 (identity filter)
        weights_[NumTaps / 2] = ComplexFloat{1.0f, 0.0f};
        head_ = 0;
    }

    void set_learning_rate(float mu) noexcept {
        mu_ = mu;
    }

    float learning_rate() const noexcept {
        return mu_;
    }

    ComplexFloat filter_sample(ComplexFloat input_sample) noexcept {
        buffer_[head_] = input_sample;
        last_output_ = ComplexFloat{0.0f, 0.0f};
        for (std::size_t i = 0; i < NumTaps; ++i) {
            std::size_t buf_idx = (head_ + NumTaps - i) % NumTaps;
            last_output_ += weights_[i].conj() * buffer_[buf_idx];
        }
        return last_output_;
    }

    void adapt(ComplexFloat desired_reference) noexcept {
        ComplexFloat error = desired_reference - last_output_;
        ComplexFloat conj_error = error.conj();
        for (std::size_t i = 0; i < NumTaps; ++i) {
            std::size_t buf_idx = (head_ + NumTaps - i) % NumTaps;
            ComplexFloat gradient = conj_error * buffer_[buf_idx];
            weights_[i] += gradient * mu_;
        }
        head_ = (head_ + 1) % NumTaps;
    }

    ComplexFloat filter_and_adapt(ComplexFloat input_sample, ComplexFloat desired_reference) noexcept {
        ComplexFloat out = filter_sample(input_sample);
        adapt(desired_reference);
        return out;
    }

    ComplexFloat filter_only(ComplexFloat input_sample) noexcept {
        ComplexFloat out = filter_sample(input_sample);
        head_ = (head_ + 1) % NumTaps;
        return out;
    }

    ComplexFloat weight(std::size_t idx) const noexcept {
        return idx < NumTaps ? weights_[idx] : ComplexFloat{};
    }

private:
    float mu_{0.05f};
    ComplexFloat weights_[NumTaps];
    ComplexFloat buffer_[NumTaps];
    ComplexFloat last_output_{0.0f, 0.0f};
    std::size_t head_{0};
};

}  // namespace tinyml::baseband
