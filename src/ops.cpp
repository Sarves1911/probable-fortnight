#include "ops.hpp"
#include <stdexcept>

void relu_inplace(Tensor &t)
{
    for (auto &value : t.data)
    {
        if (value < t.zero_point)
        {
            value = static_cast<int8_t>(t.zero_point);
        }
    }
}

void relu_inplace(FloatTensor &t)
{
    for (auto &value : t.data)
    {
        if (value < 0.0f)
        {
            value = 0.0f;
        }
    }
}

FloatTensor fc_linear(const FloatTensor &input, const Tensor &weight, const FloatTensor &bias)
{
    if (weight.shape.size() != 2)
    {
        throw std::runtime_error("fc_linear expects weight to be 2D");
    }

    int out_features = weight.shape[0];
    int in_features = weight.shape[1];
    if (input.numel() != in_features)
    {
        throw std::runtime_error("In features dont match");
    }
    if (bias.numel() != out_features)
    {
        throw std::runtime_error("Bias size does not match out_features");
    }
    FloatTensor output;
    output.shape = {out_features};
    output.data.resize(out_features);

    for (int i = 0; i < out_features; i++)
    {
        float acc = bias.data[i];
        for (int j = 0; j < in_features; j++)
        {
            float q_weight = weight.data[i * in_features + j];
            float real_weight = (q_weight - weight.zero_point) * weight.scale;
            acc += input.data[j] * real_weight;
        }
        output.data[i] = acc;
    }

    return output;
}