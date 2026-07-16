#pragma once

#include "tensor.hpp"

void relu_inplace(Tensor &t);

FloatTensor fc_linear(
    const FloatTensor &input,
    const Tensor &weight,
    const FloatTensor &bias);

void relu_inplace(FloatTensor &t);