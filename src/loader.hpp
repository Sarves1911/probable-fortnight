#pragma once

#include <string>
#include "tensor.hpp"

Tensor load_tensor_i8(const std::string &bin_path, const std::string &meta_path);