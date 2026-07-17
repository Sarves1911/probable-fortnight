# LeNet-5 C++ Inference Engine

Zero-dependency C++ inference engine for LeNet-5 on MNIST with INT8 weight quantization.

No PyTorch, no ONNX, and no ML framework at inference time — just C++ loading exported weights and running the forward pass with handwritten operators.

Current result on Apple M4:

| Metric               |              Value |
| -------------------- | -----------------: |
| Accuracy             |             98.78% |
| Correct predictions  |       9878 / 10000 |
| Total inference time |         1020.28 ms |
| Latency              |  0.102028 ms/image |
| Throughput           | 9801.21 images/sec |

---

## What this project does

This project implements the LeNet-5 inference path from scratch in C++:

```text
MNIST image
→ Conv2D
→ ReLU
→ MaxPool2D
→ Conv2D
→ ReLU
→ MaxPool2D
→ Flatten
→ Fully Connected
→ ReLU
→ Fully Connected
→ ReLU
→ Fully Connected
→ Softmax
→ Predicted digit
```

The model is trained in Python, then its weights are exported into raw binary files. The C++ program loads those binary weights and runs inference without using PyTorch, ONNX Runtime, TensorFlow, OpenCV, Eigen, or any external ML library.

---

## Motivation

Most ML engineers use inference frameworks as black boxes. This project strips that away and implements the core pieces manually:

- tensor storage and flat memory indexing
- binary weight loading
- INT8 weight quantization metadata
- Conv2D nested loops
- MaxPool2D
- fully connected layers
- ReLU
- Softmax
- accuracy evaluation
- benchmark timing
- unit tests for core math ops

The goal is to understand what happens below the framework layer: memory layout, tensor shapes, quantization scales, and the actual arithmetic behind CNN inference.

---

## Architecture

```text
Input: 1 × 28 × 28

Conv2D: 1 → 6, kernel 5 × 5, stride 1
Output: 6 × 24 × 24

ReLU

MaxPool2D: 2 × 2, stride 2
Output: 6 × 12 × 12

Conv2D: 6 → 16, kernel 5 × 5, stride 1
Output: 16 × 8 × 8

ReLU

MaxPool2D: 2 × 2, stride 2
Output: 16 × 4 × 4

Flatten
Output: 256

FC1: 256 → 120
ReLU

FC2: 120 → 84
ReLU

FC3: 84 → 10

Softmax
Output: 10 class probabilities
```

---

## Quantization scheme

This is currently a **weight-only quantized inference engine**.

Weights are quantized to INT8 during export using symmetric per-tensor quantization:

```text
q_weight = round(real_weight / scale)
```

The scale and zero point are stored in a JSON metadata file next to each binary weight file.

At inference time, weights are stored as `int8_t`, then dequantized during Conv2D and FC computation:

```text
real_weight = (q_weight - zero_point) * scale
acc += input_activation * real_weight
```

Activations and accumulators are currently `float32`.

So the current implementation is:

```text
INT8 weights + float32 activations + float32 accumulation
```

The next milestone is true fixed-point inference:

```text
INT8 activations + INT8 weights → INT32 accumulation → requantization
```

---

## Results

Measured on Apple M4, single-threaded, Release build, `-O3 -march=native`.

Dataset: full MNIST test set, 10,000 images.

| Implementation                                 | Accuracy |           Latency |         Throughput |
| ---------------------------------------------- | -------: | ----------------: | -----------------: |
| C++ engine, INT8 weights + float32 activations |   98.78% | 0.102028 ms/image | 9801.21 images/sec |

Benchmark output:

```text
Correct: 9878 / 10000
Accuracy: 98.78%
Total inference time: 1020.28 ms
Latency: 0.102028 ms/image
Throughput: 9801.21 images/sec
```

---

## Project structure

```text
.
├── CMakeLists.txt
├── README.md
├── src/
│   ├── main.cpp          # Entry point, model loading, inference evaluation
│   ├── tensor.hpp        # Tensor and FloatTensor structs
│   ├── loader.hpp        # Tensor loader declarations
│   ├── loader.cpp        # Binary weight + metadata loading
│   ├── ops.hpp           # Operator declarations
│   ├── ops.cpp           # Conv2D, MaxPool2D, FC, ReLU, Softmax
│   ├── quantize.hpp      # Quantize, dequantize, requantize helpers
│   └── benchmark.hpp     # Benchmark timing utilities
├── tests/
│   ├── test_conv.cpp     # Conv2D and MaxPool2D golden tests
│   ├── test_matmul.cpp   # FC, ReLU, Softmax tests
│   └── test_quantize.cpp # Quantization helper tests
├── python/
│   ├── export_weights.py       # Export PyTorch weights to .bin + .json
│   ├── export_test_samples.py  # Export MNIST test images/labels
│   ├── validate_cpp_output.py  # Optional validation helper
│   ├── requirements.txt
│   └── lenet5_mnist.pth        # Trained checkpoint, if included locally
├── weights/              # Generated weight files, not committed
├── data/                 # Generated MNIST binary files, not committed
└── build/                # CMake build output, not committed
```

---

## Build

Requires:

- CMake 3.16+
- C++17 compiler, such as `clang++` or `g++`

Build from the repo root:

```bash
git clone https://github.com/Sarves1911/probable-fortnight
cd probable-fortnight

cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

Run tests:

```bash
ctest --test-dir build --output-on-failure
```

Expected result:

```text
100% tests passed
```

---

## Python setup

The C++ inference runtime has no third-party dependencies.

The Python export scripts require:

```text
torch
torchvision
numpy
```

Install them with:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r python/requirements.txt
```

---

## Export weights

From the repo root:

```bash
python python/export_weights.py \
  --checkpoint python/lenet5_mnist.pth \
  --out-dir weights
```

This creates files like:

```text
weights/conv1.weight.bin
weights/conv1.weight.json
weights/conv1.bias.bin
weights/conv1.bias.json
weights/conv2.weight.bin
weights/conv2.weight.json
...
```

Weights are stored as raw `int8_t`.

Biases are stored as raw `float32`.

---

## Export MNIST test data

From the repo root:

```bash
python python/export_test_samples.py
```

This creates:

```text
data/test_images.bin
data/test_labels.bin
```

Current expected format:

```text
test_images.bin: float32, shape [10000, 1, 28, 28]
test_labels.bin: uint8, shape [10000]
```

---

## Run inference

Clean benchmark output:

```bash
./build/inference
```

Verbose per-sample predictions:

```bash
./build/inference --verbose
```

Example output:

```text
Correct: 9878 / 10000
Accuracy: 98.78%
Total inference time: 1020.28 ms
Latency: 0.102028 ms/image
Throughput: 9801.21 images/sec
```

---

## Run unit tests

```bash
ctest --test-dir build --output-on-failure
```

The test suite currently covers:

- Conv2D single-channel golden test
- Conv2D multi-channel golden test
- MaxPool2D golden test
- FC / linear layer golden test
- Float ReLU test
- Softmax probability-sum test
- Quantize/dequantize tests
- Requantization helper tests

Example:

```text
Test project build
    Start 1: test_conv
1/3 Test #1: test_conv ........................   Passed
    Start 2: test_matmul
2/3 Test #2: test_matmul ......................   Passed
    Start 3: test_quantize
3/3 Test #3: test_quantize ....................   Passed

100% tests passed, 0 tests failed out of 3
```

---

## Weight file format

Each weight tensor has a binary file and a JSON metadata file.

Example:

```text
weights/conv1.weight.bin
weights/conv1.weight.json
```

The `.bin` file stores raw row-major `int8_t` values.

For Conv2D weights, layout is:

```text
[out_channels, in_channels, kernel_height, kernel_width]
```

Example metadata:

```json
{
  "shape": [6, 1, 5, 5],
  "dtype": "int8",
  "scale": 0.003921568859368563,
  "zero_point": 0
}
```

Bias files are stored as `float32`:

```text
weights/conv1.bias.bin
weights/conv1.bias.json
```

Example bias metadata:

```json
{
  "shape": [6],
  "dtype": "float32",
  "scale": 1.0,
  "zero_point": 0
}
```

---

## Implementation notes

### Tensor layout

The engine uses channel-first layout for image tensors:

```text
channels × height × width
```

Flat index:

```cpp
index = c * height * width + h * width + w;
```

Conv2D weight layout:

```text
out_channels × in_channels × kernel_height × kernel_width
```

Flat Conv2D weight index:

```cpp
index =
    oc * in_channels * kernel_height * kernel_width
  + ic * kernel_height * kernel_width
  + kh * kernel_width
  + kw;
```

### Conv2D

Conv2D is implemented as a direct nested-loop convolution.

Loop order:

```text
output channel
→ output row
→ output column
→ input channel
→ kernel row
→ kernel column
```

This is intentionally simple and readable. No im2col, BLAS, SIMD, or framework kernels are used.

### MaxPool2D

MaxPool2D scans each non-overlapping window and keeps the maximum value.

For LeNet-5:

```text
6 × 24 × 24 → 6 × 12 × 12
16 × 8 × 8 → 16 × 4 × 4
```

### Fully connected layers

The FC layer computes:

```text
output = W * input + bias
```

Weights are stored as INT8 and dequantized before multiplication.

### Softmax

Softmax uses the numerically stable form:

```text
exp(x_i - max(x)) / sum(exp(x_j - max(x)))
```

This avoids overflow on large logits.

### Benchmarking

Benchmarking is separated into `src/benchmark.hpp`.

The timing measures inference only. Model loading and dataset loading are excluded.

Per-sample `std::cout` is disabled unless `--verbose` is passed, because printing thousands of lines significantly affects throughput.

---

## Current limitations

- Activations are still `float32`.
- Accumulation is still `float32`.
- This is not yet full integer-only inference.
- No SIMD or multithreading yet.
- No padding support in Conv2D.
- No batching support.
- JSON metadata parsing is intentionally minimal and dependency-free.

---

## Roadmap

Planned improvements:

- Implement true INT8 activations.
- Add INT8 × INT8 → INT32 Conv2D.
- Add INT8 × INT8 → INT32 FC.
- Add requantization between layers.
- Compare against PyTorch float32 and PyTorch quantized inference.
- Add GitHub Actions CI.
- Add optional im2col + GEMM backend.
- Add optional ARM NEON acceleration.
- Refactor `main.cpp` into `model.hpp/model.cpp` and `data.hpp/data.cpp`.

---

## License

MIT
