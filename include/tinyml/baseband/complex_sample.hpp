#pragma once

#include <cmath>
#include <cstdint>

namespace tinyml::baseband {

/**
 * @brief High-performance floating-point Complex Sample representation (I + jQ).
 * 
 * In wireless physical layers (5G/6G), baseband signals are represented by 
 * orthogonal In-Phase (I) and Quadrature (Q) components.
 */
struct ComplexFloat {
    float i{0.0f};  // In-Phase (Real)
    float q{0.0f};  // Quadrature (Imaginary)

    constexpr ComplexFloat() noexcept = default;
    constexpr ComplexFloat(float in_phase, float quad) noexcept : i(in_phase), q(quad) {}

    constexpr ComplexFloat operator+(const ComplexFloat& o) const noexcept {
        return {i + o.i, q + o.q};
    }

    constexpr ComplexFloat operator-(const ComplexFloat& o) const noexcept {
        return {i - o.i, q - o.q};
    }

    // Complex multiplication: (I1 + jQ1)(I2 + jQ2) = (I1*I2 - Q1*Q2) + j(I1*Q2 + Q1*I2)
    constexpr ComplexFloat operator*(const ComplexFloat& o) const noexcept {
        return {i * o.i - q * o.q, i * o.q + q * o.i};
    }

    constexpr ComplexFloat operator*(float scalar) const noexcept {
        return {i * scalar, q * scalar};
    }

    ComplexFloat& operator+=(const ComplexFloat& o) noexcept {
        i += o.i;
        q += o.q;
        return *this;
    }

    ComplexFloat& operator-=(const ComplexFloat& o) noexcept {
        i -= o.i;
        q -= o.q;
        return *this;
    }

    // Complex conjugate: I - jQ
    constexpr ComplexFloat conj() const noexcept {
        return {i, -q};
    }

    // Squared magnitude: I^2 + Q^2 (Instant power measurement without expensive sqrt)
    constexpr float mag_sq() const noexcept {
        return i * i + q * q;
    }

    float mag() const noexcept {
        return std::sqrt(mag_sq());
    }
};

/**
 * @brief Q15 Fixed-Point Complex Sample (1 sign bit, 15 fractional bits).
 * 
 * Simulates multi-core DSP hardware registers where fixed-point arithmetic
 * is used to minimize silicon area, power dissipation, and latency.
 */
struct ComplexQ15 {
    int16_t i{0};
    int16_t q{0};

    constexpr ComplexQ15() noexcept = default;
    constexpr ComplexQ15(int16_t in_phase, int16_t quad) noexcept : i(in_phase), q(quad) {}

    static constexpr int16_t float_to_q15(float v) noexcept {
        if (v >= 0.999969f) return 32767;
        if (v <= -1.0f) return -32768;
        return static_cast<int16_t>(v * 32768.0f);
    }

    static constexpr float q15_to_float(int16_t v) noexcept {
        return static_cast<float>(v) / 32768.0f;
    }

    static constexpr ComplexQ15 from_float(const ComplexFloat& cf) noexcept {
        return {float_to_q15(cf.i), float_to_q15(cf.q)};
    }

    constexpr ComplexFloat to_float() const noexcept {
        return {q15_to_float(i), q15_to_float(q)};
    }

    // Fixed-point complex multiplication with saturating shift
    ComplexQ15 operator*(const ComplexQ15& o) const noexcept {
        int32_t real_part = (static_cast<int32_t>(i) * o.i - static_cast<int32_t>(q) * o.q) >> 15;
        int32_t imag_part = (static_cast<int32_t>(i) * o.q + static_cast<int32_t>(q) * o.i) >> 15;

        auto clamp = [](int32_t v) -> int16_t {
            if (v > 32767) return 32767;
            if (v < -32768) return -32768;
            return static_cast<int16_t>(v);
        };

        return {clamp(real_part), clamp(imag_part)};
    }
};

}  // namespace tinyml::baseband
