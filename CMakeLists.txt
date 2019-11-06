cmake_minimum_required(VERSION 3.3.2)
project(Component CXX)
find_package(HPX REQUIRED)

add_hpx_executable(Stencil4 SOURCES Stencil4.cpp)
add_hpx_executable(Stencil5 SOURCES Stencil5.cpp)
