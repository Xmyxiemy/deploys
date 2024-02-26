cmake_minimum_required(VERSION 3.0.0)
set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 定义项目名称
set(PROJECT_NAME freeyolo)
project(${PROJECT_NAME})

# 自定义的包含目录
# include_directories("include/")
# Declare the executable target built from your sources
add_executable(${PROJECT_NAME} main.cpp)
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_14)


message(STATUS "========================================")
message(STATUS "               Onnx Runtime             ")
message(STATUS "========================================")
option(ONNXRUNTIME_DIR "/root/xierry/deploy/YOLO-NAS-onnxruntime/onnxruntime-linux-x64-gpu-1.15.0" STRING)
# set(USE_CUDA OFF)
set(ONNXRUNTIME_DIR /root/xierry/deploy/YOLO-NAS-onnxruntime/onnxruntime-linux-x64-gpu-1.15.0) # 手动设置
find_library(PATH ${ONNXRUNTIME_DIR})
message(STATUS "ONNXRUNTIME_DIR: ${ONNXRUNTIME_DIR}\n")

# Link your application with Onnxruntime libraries/header
target_include_directories(${PROJECT_NAME} PRIVATE "${ONNXRUNTIME_DIR}/include")
if (UNIX)
    target_link_libraries(${PROJECT_NAME} "${ONNXRUNTIME_DIR}/lib/libonnxruntime.so")
endif(UNIX)
if (WIN32)
    target_link_libraries(${PROJECT_NAME} "${ONNXRUNTIME_DIR}/lib/onnxruntime.lib")
endif(WIN32)
if (APPLE)
    target_link_libraries(${PROJECT_NAME} "${ONNXRUNTIME_DIR}/lib/libonnxruntime.dylib")
endif(APPLE)


message(STATUS "========================================")
message(STATUS "                OpenCV                  ")
message(STATUS "========================================")

option(OpenCV_DIR "/root/xierry/deploy/opencv-4.8.0/build" STRING) # set("OpenCV_DIR" "/usr/local")
set(OpenCV_DIR /root/xierry/deploy/opencv-4.8.0/build) # 手动设置
find_library(PATH ${OpenCV_DIR})
find_package(OpenCV REQUIRED)
message(STATUS "OpenCV Library Status:")
message(STATUS "   Version: ${OpenCV_VERSION}")
message(STATUS "   Libraries: ${OpenCV_LIBS}")
message(STATUS "   Include path: ${OpenCV_INCLUDE_DIRS}\n")

# Link your application with OpenCV libraries/header
target_include_directories(${PROJECT_NAME} PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})

