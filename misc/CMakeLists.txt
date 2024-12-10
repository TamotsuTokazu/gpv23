add_executable(my_test test.cpp)

target_include_directories(my_test PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(my_test PRIVATE gpv23 hexl)
target_compile_options(my_test PRIVATE -O0 -march=native -mavx2 -mavx512f -mavx512dq -Wno-unused-parameter)