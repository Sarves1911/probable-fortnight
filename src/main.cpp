#include <algorithm>
#include <exception>
#include <iostream>

#include "loader.hpp"
#include "ops.hpp"

static void print_shape(const std::string &name, const FloatTensor &t)
{
    std::cout << name << " shape: ";
    for (int dim : t.shape)
    {
        std::cout << dim << " ";
    }
    std::cout << "\n";
}

static void print_shape(const std::string &name, const Tensor &t)
{
    std::cout << name << " shape: ";
    for (int dim : t.shape)
    {
        std::cout << dim << " ";
    }
    std::cout << "\n";
}

static void print_values(const std::string &name, const FloatTensor &t)
{
    std::cout << name << ": ";
    for (float value : t.data)
    {
        std::cout << value << " ";
    }
    std::cout << "\n";
}

static float sum_values(const FloatTensor &t)
{
    float sum = 0.0f;
    for (float value : t.data)
    {
        sum += value;
    }
    return sum;
}

static int argmax(const FloatTensor &t)
{
    return static_cast<int>(
        std::max_element(t.data.begin(), t.data.end()) - t.data.begin());
}

int main()
{
    try
    {
        Tensor fc1_weight = load_tensor_i8(
            "weights/fc1.weight.bin",
            "weights/fc1.weight.json");

        FloatTensor fc1_bias = load_tensor_f32(
            "weights/fc1.bias.bin",
            "weights/fc1.bias.json");

        Tensor fc2_weight = load_tensor_i8(
            "weights/fc2.weight.bin",
            "weights/fc2.weight.json");

        FloatTensor fc2_bias = load_tensor_f32(
            "weights/fc2.bias.bin",
            "weights/fc2.bias.json");

        Tensor fc3_weight = load_tensor_i8(
            "weights/fc3.weight.bin",
            "weights/fc3.weight.json");

        FloatTensor fc3_bias = load_tensor_f32(
            "weights/fc3.bias.bin",
            "weights/fc3.bias.json");

        int input_size = fc1_weight.shape[1];

        FloatTensor input;
        input.shape = {input_size};
        input.data.resize(input_size);

        for (int i = 0; i < input_size; i++)
        {
            input.data[i] = 1.0f;
        }

        FloatTensor fc1_out = fc_linear(input, fc1_weight, fc1_bias);
        relu_inplace(fc1_out);

        FloatTensor fc2_out = fc_linear(fc1_out, fc2_weight, fc2_bias);
        relu_inplace(fc2_out);

        FloatTensor fc3_out = fc_linear(fc2_out, fc3_weight, fc3_bias);

        FloatTensor probs = softmax(fc3_out);

        int predicted_class = argmax(probs);

        std::cout << "FC stack + softmax test complete\n\n";

        print_shape("input", input);
        print_shape("fc1_out after ReLU", fc1_out);
        print_shape("fc2_out after ReLU", fc2_out);
        print_shape("fc3_out logits", fc3_out);
        print_shape("softmax probabilities", probs);

        std::cout << "\n";

        print_values("fc3 logits", fc3_out);
        print_values("softmax probabilities", probs);

        std::cout << "\n";

        std::cout << "Probability sum: " << sum_values(probs) << "\n";
        std::cout << "Predicted class: " << predicted_class << "\n";
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}