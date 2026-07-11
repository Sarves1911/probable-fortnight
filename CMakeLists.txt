 cmake_minimum_required(VERSION 3.16)
  project(lenet5_inference CXX)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native")
  
  add_executable(inference
    src/main.cpp src/loader.cpp src/ops.cpp src/quantize.cpp)
  
  # Optional: link Apple Accelerate for BLAS ceiling comparison
  # find_library(ACCELERATE Accelerate)
  # target_link_libraries(inference ${ACCELERATE})
