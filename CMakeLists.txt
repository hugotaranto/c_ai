cmake_minimum_required(VERSION 3.16)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(ai_neural_network C)

set(CMAKE_C_STANDARD 99)

file(GLOB SOURCES "src/*.c")

find_package(PkgConfig REQUIRED)
pkg_check_modules(OPENBLAS REQUIRED openblas)

add_executable(proj ${SOURCES})

target_include_directories(proj PRIVATE
  ${CMAKE_SOURCE_DIR}/inc
  ${OPENBLAS_INCLUDE_DIRS}
)

link_directories(${OPENBLAS_LIBRARY_DIRS})

find_library(OPENBLAS_LIB NAMES openblas PATHS /opt/homebrew/Cellar/openblas/0.3.29/lib)

# target_link_libraries(proj PRIVATE ${BLAS_LIB})
target_link_libraries(proj PRIVATE ${OPENBLAS_LIB})

