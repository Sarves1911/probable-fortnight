#include <exception>
#include <iostream>

#include "ops.hpp"

static void print_tensor_3d(const FloatTensor &t)
{
    int channels = t.shape[0];
    int height = t.shape[1];
    int width = t.shape[2];

    for (int c = 0; c < channels; c++)
    {
        std::cout << "channel " << c << ":\n";

        for (int h = 0; h < height; h++)
        {
            for (int w = 0; w < width; w++)
            {
                int idx = c * height * width + h * width + w;
                std::cout << t.data[idx] << " ";
            }
            std::cout << "\n";
        }
    }
}

int main()
{
    try
    {
        FloatTensor input;
        input.shape = {1, 4, 4};
        input.data = {
            1, 2, 3, 4,
            5, 6, 7, 8,
            9, 10, 11, 12,
            13, 14, 15, 16};

        FloatTensor output = maxpool2d(input, 1, 4, 4, 2, 2);

        std::cout << "Input:\n";
        print_tensor_3d(input);

        std::cout << "\nOutput:\n";
        print_tensor_3d(output);

        std::cout << "\nOutput shape: ";
        for (int dim : output.shape)
        {
            std::cout << dim << " ";
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