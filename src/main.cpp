#include <algorithm>
#include <cstdint>
#include <exception>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "loader.hpp"
#include "ops.hpp"
#include "benchmark.hpp"

struct Model
{
    Tensor conv1_weight;
    FloatTensor conv1_bias;

    Tensor conv2_weight;
    FloatTensor conv2_bias;

    Tensor fc1_weight;
    FloatTensor fc1_bias;

    Tensor fc2_weight;
    FloatTensor fc2_bias;

    Tensor fc3_weight;
    FloatTensor fc3_bias;
};

static FloatTensor flatten(const FloatTensor &input)
{
    FloatTensor output;
    output.shape = {input.numel()};
    output.data = input.data;
    return output;
}

static std::vector<float> read_float_file(const std::string &path)
{
    std::ifstream file(path, std::ios::binary);

    if (!file)
    {
        throw std::runtime_error("Could not open file: " + path);
    }

    file.seekg(0, std::ios::end);
    std::streamsize bytes = file.tellg();
    file.seekg(0, std::ios::beg);

    if (bytes % sizeof(float) != 0)
    {
        throw std::runtime_error("Float file size is not divisible by sizeof(float): " + path);
    }

    std::vector<float> data(bytes / sizeof(float));

    file.read(reinterpret_cast<char *>(data.data()), bytes);

    if (file.gcount() != bytes)
    {
        throw std::runtime_error("Short read from file: " + path);
    }

    return data;
}

static std::vector<uint8_t> read_u8_file(const std::string &path)
{
    std::ifstream file(path, std::ios::binary);

    if (!file)
    {
        throw std::runtime_error("Could not open file: " + path);
    }

    file.seekg(0, std::ios::end);
    std::streamsize bytes = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<uint8_t> data(bytes);

    file.read(reinterpret_cast<char *>(data.data()), bytes);

    if (file.gcount() != bytes)
    {
        throw std::runtime_error("Short read from file: " + path);
    }

    return data;
}

static FloatTensor get_image(
    const std::vector<float> &images,
    int image_index)
{
    const int image_size = 1 * 28 * 28;
    int start = image_index * image_size;

    if (start + image_size > static_cast<int>(images.size()))
    {
        throw std::runtime_error("Image index out of range");
    }

    FloatTensor image;
    image.shape = {1, 28, 28};
    image.data.resize(image_size);

    for (int i = 0; i < image_size; i++)
    {
        image.data[i] = images[start + i];
    }

    return image;
}

static Model load_model()
{
    Model model;

    model.conv1_weight = load_tensor_i8("weights/conv1.weight.bin", "weights/conv1.weight.json");
    model.conv1_bias = load_tensor_f32("weights/conv1.bias.bin", "weights/conv1.bias.json");

    model.conv2_weight = load_tensor_i8("weights/conv2.weight.bin", "weights/conv2.weight.json");
    model.conv2_bias = load_tensor_f32("weights/conv2.bias.bin", "weights/conv2.bias.json");

    model.fc1_weight = load_tensor_i8("weights/fc1.weight.bin", "weights/fc1.weight.json");
    model.fc1_bias = load_tensor_f32("weights/fc1.bias.bin", "weights/fc1.bias.json");

    model.fc2_weight = load_tensor_i8("weights/fc2.weight.bin", "weights/fc2.weight.json");
    model.fc2_bias = load_tensor_f32("weights/fc2.bias.bin", "weights/fc2.bias.json");

    model.fc3_weight = load_tensor_i8("weights/fc3.weight.bin", "weights/fc3.weight.json");
    model.fc3_bias = load_tensor_f32("weights/fc3.bias.bin", "weights/fc3.bias.json");

    return model;
}

static FloatTensor forward(const Model &model, const FloatTensor &input)
{
    FloatTensor conv1_out = conv2d(
        input,
        model.conv1_weight,
        model.conv1_bias,
        1,
        28,
        28,
        1);

    relu_inplace(conv1_out);

    FloatTensor pool1_out = maxpool2d(
        conv1_out,
        6,
        24,
        24,
        2,
        2);

    FloatTensor conv2_out = conv2d(
        pool1_out,
        model.conv2_weight,
        model.conv2_bias,
        6,
        12,
        12,
        1);

    relu_inplace(conv2_out);

    FloatTensor pool2_out = maxpool2d(
        conv2_out,
        16,
        8,
        8,
        2,
        2);

    FloatTensor flat = flatten(pool2_out);

    FloatTensor fc1_out = fc_linear(flat, model.fc1_weight, model.fc1_bias);
    relu_inplace(fc1_out);

    FloatTensor fc2_out = fc_linear(fc1_out, model.fc2_weight, model.fc2_bias);
    relu_inplace(fc2_out);

    FloatTensor fc3_out = fc_linear(fc2_out, model.fc3_weight, model.fc3_bias);

    return softmax(fc3_out);
}

int main(int argc, char *argv[])
{
    bool verbose = false;
    for (int i = 1; i < argc; i++)
    {
        if (std::string(argv[i]) == "--verbose")
            verbose = true;
    }

    try
    {
        Model model = load_model();

        std::vector<float> images = read_float_file("data/test_images.bin");
        std::vector<uint8_t> labels = read_u8_file("data/test_labels.bin");

        const int image_size = 1 * 28 * 28;
        int num_images = static_cast<int>(images.size()) / image_size;

        if (static_cast<int>(labels.size()) < num_images)
            throw std::runtime_error("Not enough labels for images");

        BenchmarkResult result = run_inference_benchmark(
            num_images,
            labels,
            [&](int i)
            {
                return get_image(images, i);
            },
            [&](const FloatTensor &image)
            {
                return forward(model, image);
            },
            50);

        print_benchmark_result(result);

        if (verbose)
        {
            std::cout << "\nPer-sample predictions:\n";

            for (int i = 0; i < num_images; i++)
            {
                FloatTensor image = get_image(images, i);
                FloatTensor probs = forward(model, image);

                int pred = argmax_tensor(probs);
                int label = static_cast<int>(labels[i]);

                std::cout << "sample " << i
                          << " pred=" << pred
                          << " label=" << label
                          << "\n";
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}