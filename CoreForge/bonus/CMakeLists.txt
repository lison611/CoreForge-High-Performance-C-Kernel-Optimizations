cmake_minimum_required(VERSION 3.12)
project(code_opt_bonus LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

file(GLOB KERNEL_SOURCES CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/src/kernel/*_bonus.cpp")

add_library(kernels_bonus STATIC ${KERNEL_SOURCES})
target_include_directories(kernels_bonus PUBLIC "${CMAKE_SOURCE_DIR}/include")

find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    target_link_libraries(kernels_bonus PRIVATE OpenMP::OpenMP_CXX)
endif()

target_compile_options(kernels_bonus PRIVATE
    -O3
    -march=native
    -Wall
    -g
    -fno-omit-frame-pointer
)

add_executable(run_all_bonus "${CMAKE_SOURCE_DIR}/src/main/run_all_bonus.cpp")
target_link_libraries(run_all_bonus PRIVATE kernels_bonus)

target_compile_features(run_all_bonus PRIVATE cxx_std_23)

target_include_directories(run_all_bonus PRIVATE "${CMAKE_SOURCE_DIR}/include")

target_compile_options(run_all_bonus PRIVATE
    -O3
    -march=native
    -Wall
    -g
    -fno-omit-frame-pointer
)

add_executable(single_bench "${CMAKE_SOURCE_DIR}/src/main/single_bench.cpp")
target_link_libraries(single_bench PRIVATE kernels_bonus)

target_compile_features(single_bench PRIVATE cxx_std_23)

target_include_directories(single_bench PRIVATE "${CMAKE_SOURCE_DIR}/include")

target_compile_options(single_bench PRIVATE
    -O3
    -march=native
    -Wall
    -g
    -fno-omit-frame-pointer
)
