cmake_minimum_required(VERSION 3.16)
project(tiny_ml_cpp VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# The library: all the ML code lives here
add_library(tinyml
  src/dataset.cpp
  src/linear_regression.cpp
  src/knn.cpp
  src/metrics.cpp
)
target_include_directories(tinyml PUBLIC include)
target_compile_options(tinyml PRIVATE -Wall -Wextra -Wpedantic)

# Example programs that use the library
add_executable(house_prices examples/house_prices.cpp)
target_link_libraries(house_prices PRIVATE tinyml)

add_executable(classify_points examples/classify_points.cpp)
target_link_libraries(classify_points PRIVATE tinyml)

# Tests
enable_testing()
add_executable(tinyml_tests tests/test_main.cpp)
target_link_libraries(tinyml_tests PRIVATE tinyml)
add_test(NAME tinyml_tests COMMAND tinyml_tests)
