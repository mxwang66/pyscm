cmake_minimum_required(VERSION 3.18)
project(pyscm LANGUAGES CXX)

if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the build type." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release RelWithDebInfo MinSizeRel)
endif()

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)
include(CheckIPOSupported)

pybind11_add_module(_scm_utility MODULE
  src/utility_python_bindings.cpp
  src/solver.cpp
)

target_compile_features(_scm_utility PRIVATE cxx_std_17)
target_include_directories(_scm_utility PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_options(_scm_utility PRIVATE
  $<$<AND:$<CONFIG:Release>,$<CXX_COMPILER_ID:MSVC>>:/O2>
  $<$<AND:$<CONFIG:Release>,$<OR:$<CXX_COMPILER_ID:GNU>,$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>>>:-O3>
)

check_ipo_supported(RESULT IPO_SUPPORTED OUTPUT IPO_ERROR)
if(IPO_SUPPORTED)
  set_property(TARGET _scm_utility PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
endif()

install(TARGETS _scm_utility
  LIBRARY DESTINATION pyscm
  RUNTIME DESTINATION pyscm
)
