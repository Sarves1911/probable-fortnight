#include <iostream>
#include <exception>

#include "loader.hpp"

int main(int argc, char **argv)
{
    if (argc != 3)
    {
        std::cerr << "Usage: ./inference <tensor.bin> <tensor.json>\n";
        return 1;
    }

    try
    {
        FloatTensor t = load_tensor_f32(argv[1], argv[2]);

        std::cout << "Loaded FloatTensor\n";

        std::cout << "Shape: ";
        for (int dim : t.shape)
        {
            std::cout << dim << " ";
        }
        std::cout << "\n";

        std::cout << "Num elements: " << t.numel() << "\n";

        std::cout << "First 10 values: ";
        for (int i = 0; i < 10 && i < static_cast<int>(t.data.size()); i++)
        {
            std::cout << t.data[i] << " ";
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