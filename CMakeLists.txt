cmake_minimum_required(VERSION 3.16)
project(gpv23 VERSION 1.0.0 LANGUAGES CXX)

include(FetchContent)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_QUIRED True)

# Optionally set build type if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Optionally enable benchmarks
option(ENABLE_BENCHMARKS "Enable benchmarking with Google Benchmark" ON)

FetchContent_Declare(
    intel-hexl
    GIT_REPOSITORY https://github.com/intel/hexl.git
    GIT_TAG        v1.2.5 # Specify the desired version
)

FetchContent_MakeAvailable(intel-hexl)

# Add source files
# Assuming your source files are in the 'src' directory
file(GLOB_RECURSE SOURCES "src/*.cpp")

# Add executable
add_executable(${PROJECT_NAME} ${SOURCES})
set_target_properties(${PROJECT_NAME} PROPERTIES ENABLE_EXPORTS ON)

# Include directories
target_include_directories(${PROJECT_NAME} PRIVATE ${PROJECT_SOURCE_DIR}/include)

# Find and link libraries (example using Boost)
target_link_libraries(${PROJECT_NAME} PRIVATE hexl)

# Add compiler definitions or options if needed
# target_compile_definitions(${PROJECT_NAME} PRIVATE SOME_DEFINITION)
target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra -pedantic -Werror -Ofast -march=native -mavx2 -mavx512f -mavx512dq -funroll-loops -ftree-vectorize -Wno-unused-parameter)

# Enable testing
enable_testing()

# Add the tests subdirectory
add_subdirectory(tests)

add_subdirectory(misc)

if(ENABLE_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

# Installation rules (optional)
# install(TARGETS ${PROJECT_NAME} DESTINATION bin)
