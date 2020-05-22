cmake_minimum_required(VERSION 3.12)

project(graph_filter)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_EXTENSIONS OFF)

# Pybind11
find_package(pybind11 REQUIRED)

# OpenMP
find_package(OpenMP REQUIRED)

# Eigen
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# Open3D
list(APPEND CMAKE_INSTALL_PREFIX "~/libraries/")
find_package(Open3D HINTS ${CMAKE_INSTALL_PREFIX}/lib/CMake)
list(APPEND Open3D_LIBRARIES dl)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${Open3D_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${Open3D_EXE_LINKER_FLAGS}")

include_directories(${Open3D_INCLUDE_DIRS})
link_directories(${Open3D_LIBRARY_DIRS})

pybind11_add_module(
    graph_filter
    wrapper.cpp
)
target_link_libraries(graph_filter PRIVATE ${Open3D_LIBRARIES} Eigen3::Eigen OpenMP::OpenMP_CXX)