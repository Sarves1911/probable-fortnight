#include <exception>
#include <iostream>

#include "loader.hpp"
#include "ops.hpp"

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

        if (fc1_weight.shape.size() != 2)
        {
            std::cerr << "fc1.weight is not 2D\n";
            return 1;
        }

        int in_features = fc1_weight.shape[1];

        FloatTensor input;
        input.shape = {in_features};
        input.data.resize(in_features);

        for (int i = 0; i < in_features; i++)
        {
            input.data[i] = 1.0f;
        }

        FloatTensor output = fc_linear(input, fc1_weight, fc1_bias);

        std::cout << "fc_linear test complete\n";

        std::cout << "Input shape: ";
        for (int dim : input.shape)
        {
            std::cout << dim << " ";
        }
        std::cout << "\n";

        std::cout << "Weight shape: ";
        for (int dim : fc1_weight.shape)
        {
            std::cout << dim << " ";
        }
        std::cout << "\n";

        std::cout << "Bias shape: ";
        for (int dim : fc1_bias.shape)
        {
            std::cout << dim << " ";
        }
        std::cout << "\n";

        std::cout << "Output shape: ";
        for (int dim : output.shape)
        {
            std::cout << dim << " ";
        }
        std::cout << "\n";

        std::cout << "First 10 output values: ";
        for (int i = 0; i < 10 && i < static_cast<int>(output.data.size()); i++)
        {
            std::cout << output.data[i] << " ";
        }
        std::cout << "\n";
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}