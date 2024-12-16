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

function(check_compile_flag SOURCE_FILE OUTPUT_FLAG)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        set(NATIVE_COMPILE_DEFINITIONS "/arch:AVX512")
    else()
        set(NATIVE_COMPILE_DEFINITIONS "-march=native")
    endif()

    try_run(CAN_RUN CAN_COMPILE ${CMAKE_BINARY_DIR}
        "${SOURCE_FILE}"
        COMPILE_DEFINITIONS ${NATIVE_COMPILE_DEFINITIONS}
        OUTPUT_VARIABLE TRY_COMPILE_OUTPUT
    )
    # Uncomment below to debug
    # message("TRY_COMPILE_OUTPUT ${TRY_COMPILE_OUTPUT}")
    if (CAN_COMPILE AND CAN_RUN STREQUAL 0)
        message(STATUS "Setting ${OUTPUT_FLAG}")
        add_definitions(-D${OUTPUT_FLAG})
        set(${OUTPUT_FLAG} 1 PARENT_SCOPE)
    else()
        message(STATUS "Compile flag not found: ${OUTPUT_FLAG}")
    endif()
endfunction()

check_compile_flag("${CMAKE_SOURCE_DIR}/checks/avx256.cpp" "HAS_AVX256")
check_compile_flag("${CMAKE_SOURCE_DIR}/checks/avx512dq.cpp" "HAS_AVX512DQ")
check_compile_flag("${CMAKE_SOURCE_DIR}/checks/avx512ifma.cpp" "HAS_AVX512IFMA")

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
target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra -pedantic -Werror -Ofast -march=native -funroll-loops -ftree-vectorize -Wno-unused-parameter)
# target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra -pedantic -Werror -Ofast -march=native -mavx512f -mavx512dq -mavx512ifma -funroll-loops -ftree-vectorize -Wno-unused-parameter)

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
