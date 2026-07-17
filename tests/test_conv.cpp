#include <cmath>
#include <exception>
#include <iostream>
#include <string>
#include <vector>

#include "ops.hpp"

static bool expect_near(float actual, float expected, float eps, const std::string &name)
{
    if (std::fabs(actual - expected) > eps)
    {
        std::cerr << "FAIL " << name
                  << ": expected " << expected
                  << ", got " << actual
                  << "\n";
        return false;
    }

    return true;
}

static bool expect_shape(const FloatTensor &t, const std::vector<int> &expected, const std::string &name)
{
    if (t.shape != expected)
    {
        std::cerr << "FAIL " << name << " shape: expected ";

        for (int dim : expected)
        {
            std::cerr << dim << " ";
        }

        std::cerr << "got ";

        for (int dim : t.shape)
        {
            std::cerr << dim << " ";
        }

        std::cerr << "\n";
        return false;
    }

    return true;
}

static bool test_conv2d_single_channel()
{
    FloatTensor input;
    input.shape = {1, 3, 3};
    input.data = {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9};

    Tensor weight;
    weight.shape = {1, 1, 2, 2};
    weight.data = {
        1, 1,
        1, 1};
    weight.scale = 1.0f;
    weight.zero_point = 0;

    FloatTensor bias;
    bias.shape = {1};
    bias.data = {0.0f};

    FloatTensor output = conv2d(input, weight, bias, 1, 3, 3, 1);

    bool ok = true;

    ok &= expect_shape(output, {1, 2, 2}, "single-channel conv");

    std::vector<float> expected = {
        12, 16,
        24, 28};

    for (int i = 0; i < static_cast<int>(expected.size()); i++)
    {
        ok &= expect_near(output.data[i], expected[i], 1e-5f, "single-channel conv value");
    }

    return ok;
}

static bool test_conv2d_multi_channel()
{
    FloatTensor input;
    input.shape = {2, 2, 2};

    // Channel 0:
    // 1 2
    // 3 4
    //
    // Channel 1:
    // 10 20
    // 30 40
    input.data = {
        1, 2,
        3, 4,

        10, 20,
        30, 40};

    Tensor weight;
    weight.shape = {1, 2, 1, 1};

    // out_ch=0, in_ch=0 weight = 2
    // out_ch=0, in_ch=1 weight = -1
    weight.data = {2, -1};
    weight.scale = 1.0f;
    weight.zero_point = 0;

    FloatTensor bias;
    bias.shape = {1};
    bias.data = {1.0f};

    FloatTensor output = conv2d(input, weight, bias, 2, 2, 2, 1);

    bool ok = true;

    ok &= expect_shape(output, {1, 2, 2}, "multi-channel conv");

    // output = bias + 2 * channel0 - channel1
    std::vector<float> expected = {
        -7, -15,
        -23, -31};

    for (int i = 0; i < static_cast<int>(expected.size()); i++)
    {
        ok &= expect_near(output.data[i], expected[i], 1e-5f, "multi-channel conv value");
    }

    return ok;
}

static bool test_maxpool2d()
{
    FloatTensor input;
    input.shape = {1, 4, 4};
    input.data = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16};

    FloatTensor output = maxpool2d(input, 1, 4, 4, 2, 2);

    bool ok = true;

    ok &= expect_shape(output, {1, 2, 2}, "maxpool");

    std::vector<float> expected = {
        6, 8,
        14, 16};

    for (int i = 0; i < static_cast<int>(expected.size()); i++)
    {
        ok &= expect_near(output.data[i], expected[i], 1e-5f, "maxpool value");
    }

    return ok;
}

int main()
{
    try
    {
        bool ok = true;

        ok &= test_conv2d_single_channel();
        ok &= test_conv2d_multi_channel();
        ok &= test_maxpool2d();

        if (!ok)
        {
            std::cerr << "test_conv failed\n";
            return 1;
        }

        std::cout << "test_conv passed\n";
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "test_conv exception: " << e.what() << "\n";
        return 1;
    }
}