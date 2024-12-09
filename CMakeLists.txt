cmake_minimum_required(VERSION 3.16)
project(gpv23 VERSION 1.0.0 LANGUAGES CXX)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_QUIRED True)

# Optionally set build type if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Optionally enable benchmarks
option(ENABLE_BENCHMARKS "Enable benchmarking with Google Benchmark" ON)

# Add source files
# Assuming your source files are in the 'src' directory
file(GLOB_RECURSE SOURCES "src/*.cpp")

# Add executable
add_executable(${PROJECT_NAME} ${SOURCES})
set_target_properties(${PROJECT_NAME} PROPERTIES ENABLE_EXPORTS ON)

# Include directories
target_include_directories(${PROJECT_NAME} PRIVATE ${PROJECT_SOURCE_DIR}/include)

# Find and link libraries (example using Boost)
# find_package(Boost 1.71 REQUIRED COMPONENTS filesystem system)
# if(Boost_FOUND)
#     target_include_directories(${PROJECT_NAME} PRIVATE ${Boost_INCLUDE_DIRS})
#     target_link_libraries(${PROJECT_NAME} PRIVATE ${Boost_LIBRARIES})
# endif()

# Add compiler definitions or options if needed
# target_compile_definitions(${PROJECT_NAME} PRIVATE SOME_DEFINITION)
target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra -pedantic -Werror -Ofast -march=native -mavx2 -funroll-loops -ftree-vectorize -Wno-unused-parameter)

# Enable testing
enable_testing()

# Add the tests subdirectory
add_subdirectory(tests)

if(ENABLE_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

# Installation rules (optional)
# install(TARGETS ${PROJECT_NAME} DESTINATION bin)
