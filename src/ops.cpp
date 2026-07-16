#include "ops.hpp"
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <limits>

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

FloatTensor softmax(const FloatTensor &logits)
{
    if (logits.data.empty())
    {
        throw std::runtime_error("softmax received empty logits");
    }

    FloatTensor output;
    output.shape = logits.shape;
    output.data.resize(logits.data.size());

    float max_val = *std::max_element(logits.data.begin(), logits.data.end());

    float sum = 0.0f;

    for (int i = 0; i < static_cast<int>(logits.data.size()); i++)
    {
        output.data[i] = std::exp(logits.data[i] - max_val);
        sum += output.data[i];
    }

    if (sum == 0.0f)
    {
        throw std::runtime_error("softmax sum is zero");
    }

    for (int i = 0; i < static_cast<int>(output.data.size()); i++)
    {
        output.data[i] /= sum;
    }

    return output;
}

FloatTensor maxpool2d(
    const FloatTensor &input,
    int channels,
    int height,
    int width,
    int pool_size,
    int stride)
{
    if (input.numel() != channels * height * width)
    {
        throw std::runtime_error("size does not match");
    }
    if (pool_size <= 0)
    {
        throw std::runtime_error("Pool size is less than 0");
    }
    if (stride <= 0)
    {
        throw std::runtime_error("Stride is less than 0");
    }
    if (height < pool_size || width < pool_size)
    {
        throw std::runtime_error("Pool size is larger than input dimensions");
    }

    int out_height = (height - pool_size) / stride + 1;
    int out_width = (width - pool_size) / stride + 1;

    FloatTensor output;
    output.shape = {channels, out_height, out_width};

    output.data.resize(channels * out_height * out_width);

    for (int c = 0; c < channels; c++)
    {
        for (int oh = 0; oh < out_height; oh++)
        {
            for (int ow = 0; ow < out_width; ow++)
            {
                float max_value = std::numeric_limits<float>::lowest();

                for (int kh = 0; kh < pool_size; kh++)
                {
                    for (int kw = 0; kw < pool_size; kw++)
                    {
                        int input_h = oh * stride + kh;
                        int input_w = ow * stride + kw;

                        int input_index = c * height * width + input_h * width + input_w;

                        max_value = std::max(max_value, input.data[input_index]);
                    }
                }

                int output_index = c * out_height * out_width + oh * out_width + ow;

                output.data[output_index] = max_value;
            }
        }
    }
    return output;
}