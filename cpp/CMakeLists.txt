cmake_minimum_required(VERSION 3.18)
project(pyscm LANGUAGES CXX)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_scm_utility MODULE
  src/utility_python_bindings.cpp
  src/solver.cpp
)

target_compile_features(_scm_utility PRIVATE cxx_std_14)
target_include_directories(_scm_utility PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

install(TARGETS _scm_utility
  LIBRARY DESTINATION pyscm
  RUNTIME DESTINATION pyscm
)
