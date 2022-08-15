cmake_minimum_required(VERSION 3.16)
project(EML)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -O3  -Wall -Wextra -std=c++20")
if (NOT $ENV{WORKING_DIR})
    SET(WORKING_DIR ../../../)
else ()
    SET(WORKING_DIR $ENV{WORKING_DIR})
endif ()


configure_file(embedded_implementation/helpers/globals.h.in generated/globals.h)

include_directories(${CMAKE_BINARY_DIR}/generated/)

add_subdirectory(embedded_implementation/demo)
