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
        std::cerr << "FAIL " << name << " shape\n";
        return false;
    }

    return true;
}

static bool test_fc_linear()
{
    FloatTensor input;
    input.shape = {3};
    input.data = {1.0f, 2.0f, -1.0f};

    Tensor weight;
    weight.shape = {2, 3};

    // scale = 0.5
    // row 0 real weights:  1, -2,  3
    // row 1 real weights: .5,  0, -1
    weight.data = {
        2, -4, 6,
        1, 0, -2};

    weight.scale = 0.5f;
    weight.zero_point = 0;

    FloatTensor bias;
    bias.shape = {2};
    bias.data = {0.25f, -0.5f};

    FloatTensor output = fc_linear(input, weight, bias);

    bool ok = true;

    ok &= expect_shape(output, {2}, "fc_linear");

    // out0 = 0.25 + 1*1 + 2*(-2) + (-1)*3 = -5.75
    // out1 = -0.5 + 1*0.5 + 2*0 + (-1)*(-1) = 1.0
    ok &= expect_near(output.data[0], -5.75f, 1e-5f, "fc output 0");
    ok &= expect_near(output.data[1], 1.0f, 1e-5f, "fc output 1");

    return ok;
}

static bool test_relu_float()
{
    FloatTensor t;
    t.shape = {5};
    t.data = {-2.0f, -0.1f, 0.0f, 3.0f, 5.0f};

    relu_inplace(t);

    std::vector<float> expected = {0.0f, 0.0f, 0.0f, 3.0f, 5.0f};

    bool ok = true;

    for (int i = 0; i < static_cast<int>(expected.size()); i++)
    {
        ok &= expect_near(t.data[i], expected[i], 1e-6f, "relu float");
    }

    return ok;
}

static bool test_softmax()
{
    FloatTensor logits;
    logits.shape = {3};
    logits.data = {1.0f, 2.0f, 3.0f};

    FloatTensor probs = softmax(logits);

    bool ok = true;

    ok &= expect_shape(probs, {3}, "softmax");

    float sum = 0.0f;
    for (float value : probs.data)
    {
        sum += value;
    }

    ok &= expect_near(sum, 1.0f, 1e-5f, "softmax sum");

    if (!(probs.data[2] > probs.data[1] && probs.data[1] > probs.data[0]))
    {
        std::cerr << "FAIL softmax ordering\n";
        ok = false;
    }

    return ok;
}

int main()
{
    try
    {
        bool ok = true;

        ok &= test_fc_linear();
        ok &= test_relu_float();
        ok &= test_softmax();

        if (!ok)
        {
            std::cerr << "test_matmul failed\n";
            return 1;
        }

        std::cout << "test_matmul passed\n";
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "test_matmul exception: " << e.what() << "\n";
        return 1;
    }
}