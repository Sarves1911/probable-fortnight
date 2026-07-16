#include "ops.hpp"

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