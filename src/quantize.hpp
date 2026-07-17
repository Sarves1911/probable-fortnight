#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>

inline int8_t clamp_i8(int32_t value)
{
    value = std::clamp(value, -128, 127);
    return static_cast<int8_t>(value);
}

inline float dequantize_i8(int8_t q, float scale, int zero_point)
{
    return (static_cast<int>(q) - zero_point) * scale;
}

inline int8_t quantize_to_i8(float value, float scale, int zero_point)
{
    if (scale <= 0.0f)
    {
        throw std::runtime_error("quantize_to_i8 requires positive scale");
    }

    int32_t q = static_cast<int32_t>(std::round(value / scale)) + zero_point;
    return clamp_i8(q);
}

inline int8_t requantize_i32_to_i8(
    int32_t acc,
    float input_scale,
    float weight_scale,
    float output_scale,
    int output_zero_point)
{
    if (input_scale <= 0.0f || weight_scale <= 0.0f || output_scale <= 0.0f)
    {
        throw std::runtime_error("requantize_i32_to_i8 requires positive scales");
    }

    float multiplier = (input_scale * weight_scale) / output_scale;
    int32_t q = static_cast<int32_t>(std::round(acc * multiplier)) + output_zero_point;

    return clamp_i8(q);
}