#include <iostream>
#include "loader.hpp"
#include "ops.hpp"

int main(int argc, char **argv)
{
    if (argc != 3)
    {
        std::cerr << "Usage: ./inference <tensor.bin> <tensor.json>\n";
        return 1;
    }

    try
    {
        Tensor t = load_tensor_i8(argv[1], argv[2]);

        std::cout << "Loaded tensor\n";

        std::cout << "Shape: ";
        for (int dim : t.shape)
        {
            std::cout << dim << " ";
        }
        std::cout << "\n";

        std::cout << "Scale: " << t.scale << "\n";
        std::cout << "Zero point: " << t.zero_point << "\n";

        std::cout << "First 10 values: ";
        for (int i = 0; i < 10 && i < static_cast<int>(t.data.size()); i++)
        {
            std::cout << static_cast<int>(t.data[i]) << " ";
        }
        std::cout << "\n";

        relu_inplace(t);

        std::cout << "First 10 values after ReLU: ";
        for (int i = 0; i < 10 && i < static_cast<int>(t.data.size()); i++)
        {
            std::cout << static_cast<int>(t.data[i]) << " ";
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