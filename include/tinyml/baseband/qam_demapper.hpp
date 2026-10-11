#pragma once

#include "tinyml/baseband/complex_sample.hpp"
#include <array>
#include <cmath>
#include <cstdint>

namespace tinyml::baseband {

/**
 * @brief 16-QAM Modulation and Nearest-Neighbor Demapping.
 * 
 * In cellular basebands (5G NR / 6G), high data rates are achieved by 
 * modulating bits into 16-QAM constellations (4 bits per complex symbol).
 * 
 * At the receiver, the demapper uses Nearest Neighbor spatial classification 
 * to decode continuous noisy analog IQ energy back into digital bits.
 */
class QAM16 {
public:
    // Ideal 16-QAM grid coordinates normalized to unit average energy:
    // Levels: {-3, -1, +1, +3} / sqrt(10)
    static constexpr float SCALE = 0.316227766f; // 1 / sqrt(10)

    /**
     * @brief Maps 4 bits (0..15) to a normalized complex IQ constellation symbol.
     */
    static constexpr ComplexFloat modulate(uint8_t nibble) noexcept {
        nibble &= 0x0F;
        // Gray-coded mapping for minimal bit-error rate
        float i_val = (nibble & 0x02) ? ((nibble & 0x01) ? -3.0f : -1.0f)
                                      : ((nibble & 0x01) ?  3.0f :  1.0f);
        float q_val = (nibble & 0x08) ? ((nibble & 0x04) ? -3.0f : -1.0f)
                                      : ((nibble & 0x04) ?  3.0f :  1.0f);
        return {i_val * SCALE, q_val * SCALE};
    }

    /**
     * @brief Nearest-Neighbor Constellation Slicer / Hard Demapper.
     * 
     * Performs instant O(1) decision thresholding to classify the received 
     * noisy IQ coordinate into the closest constellation point and decoded bits.
     */
    static uint8_t demap(ComplexFloat sample, ComplexFloat* closest_ideal = nullptr) noexcept {
        // Threshold slicing across [-2, 0, +2] * SCALE
        float norm_i = sample.i / SCALE;
        float norm_q = sample.q / SCALE;

        float sliced_i = (norm_i > 2.0f) ? 3.0f : ((norm_i > 0.0f) ? 1.0f : ((norm_i > -2.0f) ? -1.0f : -3.0f));
        float sliced_q = (norm_q > 2.0f) ? 3.0f : ((norm_q > 0.0f) ? 1.0f : ((norm_q > -2.0f) ? -1.0f : -3.0f));

        if (closest_ideal) {
            *closest_ideal = ComplexFloat{sliced_i * SCALE, sliced_q * SCALE};
        }

        // Reconstruct Gray-coded nibble
        uint8_t nibble = 0;
        if (sliced_i == 3.0f)  nibble |= 0x01;
        else if (sliced_i == 1.0f) nibble |= 0x00;
        else if (sliced_i == -1.0f) nibble |= 0x02;
        else if (sliced_i == -3.0f) nibble |= 0x03;

        if (sliced_q == 3.0f)  nibble |= 0x04;
        else if (sliced_q == 1.0f) nibble |= 0x00;
        else if (sliced_q == -1.0f) nibble |= 0x08;
        else if (sliced_q == -3.0f) nibble |= 0x0C;

        return nibble;
    }

    /**
     * @brief Computes 3GPP Error Vector Magnitude (EVM) in percentage.
     * 
     * EVM is the core telecom physical-layer metric measuring received 
     * radio signal fidelity: RMS error vector power relative to reference signal power.
     */
    static float compute_evm_rms(const ComplexFloat* received, const ComplexFloat* reference, std::size_t count) noexcept {
        if (count == 0) return 0.0f;
        float error_power = 0.0f;
        float ref_power = 0.0f;

        for (std::size_t i = 0; i < count; ++i) {
            ComplexFloat err = received[i] - reference[i];
            error_power += err.mag_sq();
            ref_power += reference[i].mag_sq();
        }

        if (ref_power <= 0.0f) return 0.0f;
        return std::sqrt(error_power / ref_power) * 100.0f;
    }
};

}  // namespace tinyml::baseband
