#pragma once

#include <cstdint>
#include <vector>
#include <stdexcept>

struct Tensor
{
    std::vector<int> shape;
    std::vector<int8_t> data;
    float scale = 1.0f;
    int zero_point = 0;

    int numel() const
    {
        int total = 1;
        for (int dim : shape)
        {
            total *= dim;
        }
        return total;
    }

    int index4d(int n, int c, int h, int w) const
    {
        if (shape.size() != 4)
        {
            throw std::runtime_error("index4d called on non-4D tensor");
        }

        return n * shape[1] * shape[2] * shape[3] + c * shape[2] * shape[3] + h * shape[3] + w;
    }

    int8_t at(int n, int c, int h, int w) const
    {
        return data[index4d(n, c, h, w)];
    }

    int8_t &at(int n, int c, int h, int w)
    {
        return data[index4d(n, c, h, w)];
    }
};

struct FloatTensor
{
    std::vector<int> shape;
    std::vector<float> data;

    int numel() const
    {
        int total = 1;
        for (int dim : shape)
        {
            total *= dim;
        }
        return total;
    }
};