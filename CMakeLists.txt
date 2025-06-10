cmake_minimum_required(VERSION 3.16)

project(ai_neural_network C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Default to Release if no build type is set
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
endif()

# Set compile flags for different build types
set(CMAKE_C_FLAGS_DEBUG "-g")
set(CMAKE_C_FLAGS_RELEASE "-O3 -DNDEBUG")

# Source files
file(GLOB SOURCES "src/*.c")

# Find OpenBLAS
find_package(PkgConfig REQUIRED)
pkg_check_modules(OPENBLAS REQUIRED openblas)

# Add executable
add_executable(proj ${SOURCES})

# Include directories
target_include_directories(proj PRIVATE
  ${CMAKE_SOURCE_DIR}/inc
  ${OPENBLAS_INCLUDE_DIRS}
)

# Link OpenBLAS
link_directories(${OPENBLAS_LIBRARY_DIRS})
find_library(OPENBLAS_LIB NAMES openblas PATHS /opt/homebrew/Cellar/openblas/0.3.29/lib)
target_link_libraries(proj PRIVATE ${OPENBLAS_LIB})

# Enable AddressSanitizer only in Debug mode
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  target_compile_options(proj PRIVATE -fsanitize=address)
  target_link_options(proj PRIVATE -fsanitize=address)
endif()
