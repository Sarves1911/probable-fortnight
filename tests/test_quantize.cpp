#include <cmath>
#include <cstdint>
#include <exception>
#include <iostream>
#include <string>

#include "quantize.hpp"

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

static bool expect_i8(int8_t actual, int expected, const std::string &name)
{
    int actual_int = static_cast<int>(actual);

    if (actual_int != expected)
    {
        std::cerr << "FAIL " << name
                  << ": expected " << expected
                  << ", got " << actual_int
                  << "\n";
        return false;
    }

    return true;
}

static bool test_dequantize()
{
    bool ok = true;

    ok &= expect_near(dequantize_i8(static_cast<int8_t>(10), 0.1f, 0), 1.0f, 1e-6f, "dequantize zp0");
    ok &= expect_near(dequantize_i8(static_cast<int8_t>(15), 0.2f, 5), 2.0f, 1e-6f, "dequantize zp5");

    return ok;
}

static bool test_quantize()
{
    bool ok = true;

    ok &= expect_i8(quantize_to_i8(1.0f, 0.1f, 0), 10, "quantize 1.0");
    ok &= expect_i8(quantize_to_i8(-1.0f, 0.1f, 0), -10, "quantize -1.0");
    ok &= expect_i8(quantize_to_i8(1.0f, 0.2f, 5), 10, "quantize with zp");
    ok &= expect_i8(quantize_to_i8(1000.0f, 0.1f, 0), 127, "quantize upper clamp");
    ok &= expect_i8(quantize_to_i8(-1000.0f, 0.1f, 0), -128, "quantize lower clamp");

    return ok;
}

static bool test_requantize()
{
    bool ok = true;

    // multiplier = input_scale * weight_scale / output_scale
    // multiplier = 0.02 * 0.05 / 0.1 = 0.01
    // q = round(100 * 0.01) + 0 = 1
    ok &= expect_i8(
        requantize_i32_to_i8(100, 0.02f, 0.05f, 0.1f, 0),
        1,
        "requantize zp0");

    // same but output zero point = 5
    ok &= expect_i8(
        requantize_i32_to_i8(100, 0.02f, 0.05f, 0.1f, 5),
        6,
        "requantize zp5");

    ok &= expect_i8(
        requantize_i32_to_i8(1000000, 1.0f, 1.0f, 0.001f, 0),
        127,
        "requantize upper clamp");

    ok &= expect_i8(
        requantize_i32_to_i8(-1000000, 1.0f, 1.0f, 0.001f, 0),
        -128,
        "requantize lower clamp");

    return ok;
}

static bool test_invalid_scale()
{
    bool ok = true;

    try
    {
        (void)quantize_to_i8(1.0f, 0.0f, 0);
        std::cerr << "FAIL quantize invalid scale did not throw\n";
        ok = false;
    }
    catch (const std::exception &)
    {
        // expected
    }

    try
    {
        (void)requantize_i32_to_i8(10, 1.0f, 0.0f, 1.0f, 0);
        std::cerr << "FAIL requantize invalid scale did not throw\n";
        ok = false;
    }
    catch (const std::exception &)
    {
        // expected
    }

    return ok;
}

int main()
{
    try
    {
        bool ok = true;

        ok &= test_dequantize();
        ok &= test_quantize();
        ok &= test_requantize();
        ok &= test_invalid_scale();

        if (!ok)
        {
            std::cerr << "test_quantize failed\n";
            return 1;
        }

        std::cout << "test_quantize passed\n";
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "test_quantize exception: " << e.what() << "\n";
        return 1;
    }
}