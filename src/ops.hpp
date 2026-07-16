#pragma once

#include "tensor.hpp"

void relu_inplace(Tensor &t);

FloatTensor fc_linear(
    const FloatTensor &input,
    const Tensor &weight,
    const FloatTensor &bias);

void relu_inplace(FloatTensor &t);

FloatTensor softmax(const FloatTensor &logits);

FloatTensor maxpool2d(
    const FloatTensor &input,
    int channels,
    int height,
    int width,
    int pool_size,
    int stride);