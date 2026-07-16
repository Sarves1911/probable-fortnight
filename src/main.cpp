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

static void print_first_values(const std::string &name, const FloatTensor &t, int count)
{
    std::cout << name << " first values: ";
    for (int i = 0; i < count && i < static_cast<int>(t.data.size()); i++)
    {
        std::cout << t.data[i] << " ";
    }
    std::cout << "\n";
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

        std::cout << "FC stack test complete\n\n";

        print_shape("input", input);
        print_shape("fc1.weight", fc1_weight);
        print_shape("fc1.bias", fc1_bias);
        print_shape("fc1_out after ReLU", fc1_out);

        std::cout << "\n";

        print_shape("fc2.weight", fc2_weight);
        print_shape("fc2.bias", fc2_bias);
        print_shape("fc2_out after ReLU", fc2_out);

        std::cout << "\n";

        print_shape("fc3.weight", fc3_weight);
        print_shape("fc3.bias", fc3_bias);
        print_shape("fc3_out logits", fc3_out);

        std::cout << "\n";

        print_first_values("fc1_out", fc1_out, 10);
        print_first_values("fc2_out", fc2_out, 10);
        print_first_values("fc3_out logits", fc3_out, 10);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}