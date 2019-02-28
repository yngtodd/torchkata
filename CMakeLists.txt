cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

project (torchkata-kata)

find_package(Torch REQUIRED)

# We have gtest/gmock as a Git submodule, add subdirectory to pick
# this up we we can use it as a dependency
add_subdirectory(lib/gtest)

# For the sake of form, and to give an idea of the sort of setup you'd
# find in a "real world" project, we're keeping the sources separate,
# and creating a separate archive
file (GLOB productionSources src/*.cpp)
add_library (torchkata STATIC ${productionSources})
target_include_directories (torchkata PRIVATE ${PROJECT_SOURCE_DIR}/include)

file (GLOB testSources test/*.cpp)
add_executable (torchkatatest ${testSources})
target_link_libraries (torchkatatest torchkata gmock gmock_main gtest pthread)
target_include_directories (torchkatatest PRIVATE ${gtest_SOURCE_DIR}/include)
target_include_directories (torchkatatest PRIVATE ${gmock_SOURCE_DIR}/include)
target_include_directories (torchkatatest PRIVATE ${PROJECT_SOURCE_DIR}/include)

target_link_libraries(torchkata "${TORCH_LIBRARIES}")
set_property(TARGET torchkata PROPERTY CXX_STANDARD 11)

# This enables the "make test" target, although for TDD it's likely
# you'll want to use the "make unit" target.
enable_testing ()

# Add the unit tests executable 
add_test (Unit torchkatatest)


# Generate the tags file
file (GLOB headers ${PROJECT_SOURCE_DIR}/include/*.hpp)
set_source_files_properties(TAGS PROPERTIES GENERATED true)
add_custom_target(tags
  COMMAND etags --language=c++ --declarations --ignore-indentation
  ${productionSources} ${testSources} ${headers}
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR})


# And a custom target as a convenience for running "make unit" as a
# specific target to support the TDD cycle
add_custom_target(unit
  COMMAND ${CMAKE_BINARY_DIR}/torchkatatest --gtest_color=true
   DEPENDS torchkatatest tags )