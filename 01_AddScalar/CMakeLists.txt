cmake_minimum_required(VERSION 3.10)
project(demo LANGUAGES CXX CUDA)

add_definitions(-std=c++17)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)


# cuda
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)
# tensorrt
include_directories(/home/sfy/SFY/camera/TensorRT-8.6.1.6/include)
link_directories(/home/sfy/SFY/camera/TensorRT-8.6.1.6/lib)


add_library(AddScalarPlugin SHARED AddScalarPlugin.cu)
target_link_libraries(AddScalarPlugin nvinfer cudart)