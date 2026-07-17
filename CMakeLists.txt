cmake_minimum_required(VERSION 3.16)

project(lenet5_inference CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(lenet5_core
    src/loader.cpp
    src/ops.cpp
)

target_include_directories(lenet5_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_compile_options(lenet5_core PRIVATE
    $<$<CONFIG:Release>:-O3>
    $<$<CONFIG:Release>:-march=native>
)

add_executable(inference
    src/main.cpp
)

target_link_libraries(inference PRIVATE
    lenet5_core
)

target_compile_options(inference PRIVATE
    $<$<CONFIG:Release>:-O3>
    $<$<CONFIG:Release>:-march=native>
)

enable_testing()

add_executable(test_conv
    tests/test_conv.cpp
)

target_link_libraries(test_conv PRIVATE
    lenet5_core
)

add_test(NAME test_conv COMMAND test_conv)

add_executable(test_matmul
    tests/test_matmul.cpp
)

target_link_libraries(test_matmul PRIVATE
    lenet5_core
)

add_test(NAME test_matmul COMMAND test_matmul)

add_executable(test_quantize
    tests/test_quantize.cpp
)

target_link_libraries(test_quantize PRIVATE
    lenet5_core
)

add_test(NAME test_quantize COMMAND test_quantize)