#pragma once

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "tensor.hpp"

struct BenchmarkResult
{
    int correct = 0;
    int total = 0;

    double total_ms = 0.0;
    double latency_ms_per_image = 0.0;
    double throughput_images_per_sec = 0.0;

    float accuracy = 0.0f;
};

class Timer
{
public:
    Timer()
        : start_(std::chrono::high_resolution_clock::now())
    {
    }

    double elapsed_ms() const
    {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - start_).count();
    }

private:
    std::chrono::high_resolution_clock::time_point start_;
};

inline int argmax_tensor(const FloatTensor &t)
{
    if (t.data.empty())
    {
        throw std::runtime_error("argmax_tensor received empty tensor");
    }

    return static_cast<int>(
        std::max_element(t.data.begin(), t.data.end()) - t.data.begin());
}

template <typename GetImageFn, typename ForwardFn>
BenchmarkResult run_inference_benchmark(
    int num_images,
    const std::vector<uint8_t> &labels,
    GetImageFn get_image,
    ForwardFn forward,
    int warmup_count = 50)
{
    if (num_images <= 0)
    {
        throw std::runtime_error("run_inference_benchmark requires num_images > 0");
    }

    if (static_cast<int>(labels.size()) < num_images)
    {
        throw std::runtime_error("Not enough labels for benchmark");
    }

    int actual_warmup_count = std::min(warmup_count, num_images);

    for (int i = 0; i < actual_warmup_count; i++)
    {
        FloatTensor image = get_image(i);
        FloatTensor output = forward(image);
        (void)output;
    }

    int correct = 0;

    Timer timer;

    for (int i = 0; i < num_images; i++)
    {
        FloatTensor image = get_image(i);
        FloatTensor output = forward(image);

        int pred = argmax_tensor(output);
        int label = static_cast<int>(labels[i]);

        if (pred == label)
        {
            correct++;
        }
    }

    double total_ms = timer.elapsed_ms();

    BenchmarkResult result;
    result.correct = correct;
    result.total = num_images;
    result.total_ms = total_ms;
    result.latency_ms_per_image = total_ms / static_cast<double>(num_images);
    result.throughput_images_per_sec = 1000.0 / result.latency_ms_per_image;
    result.accuracy = static_cast<float>(correct) / static_cast<float>(num_images);

    return result;
}

inline void print_benchmark_result(const BenchmarkResult &result)
{
    std::cout << "Correct: " << result.correct << " / " << result.total << "\n";
    std::cout << "Accuracy: " << result.accuracy * 100.0f << "%\n";
    std::cout << "Total inference time: " << result.total_ms << " ms\n";
    std::cout << "Latency: " << result.latency_ms_per_image << " ms/image\n";
    std::cout << "Throughput: " << result.throughput_images_per_sec << " images/sec\n";
}