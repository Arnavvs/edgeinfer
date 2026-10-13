cmake_minimum_required(VERSION 3.16)
project(edgeinfer CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# -march=native by default: the engine is built on the machine that runs it.
# EI_ARCH=haswell gives a portable AVX2+FMA+F16C build.
set(EI_ARCH "native" CACHE STRING "value for -march")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
add_compile_options(-march=${EI_ARCH} -mtune=${EI_ARCH} -fno-math-errno -Wall -Wextra -Wno-unused-parameter)

add_library(edgeinfer SHARED engine/quant.cpp engine/model.cpp engine/sampler.cpp)
target_include_directories(edgeinfer PUBLIC engine)
find_package(Threads REQUIRED)
target_link_libraries(edgeinfer PRIVATE Threads::Threads)

add_executable(ei-bench engine/bench_main.cpp)
target_link_libraries(ei-bench PRIVATE edgeinfer)
