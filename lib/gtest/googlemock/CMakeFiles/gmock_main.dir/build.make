# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.12

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /home/ygx/dev/kassa/anaconda3/lib/python3.6/site-packages/cmake/data/bin/cmake

# The command to remove a file.
RM = /home/ygx/dev/kassa/anaconda3/lib/python3.6/site-packages/cmake/data/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/ygx/src/develop/torchkata

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/ygx/src/develop/torchkata

# Include any dependencies generated for this target.
include lib/gtest/googlemock/CMakeFiles/gmock_main.dir/depend.make

# Include the progress variables for this target.
include lib/gtest/googlemock/CMakeFiles/gmock_main.dir/progress.make

# Include the compile flags for this target's objects.
include lib/gtest/googlemock/CMakeFiles/gmock_main.dir/flags.make

lib/gtest/googlemock/CMakeFiles/gmock_main.dir/__/googletest/src/gtest-all.cc.o: lib/gtest/googlemock/CMakeFiles/gmock_main.dir/flags.make
lib/gtest/googlemock/CMakeFiles/gmock_main.dir/__/googletest/src/gtest-all.cc.o: lib/gtest/googletest/src/gtest-all.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ygx/src/develop/torchkata/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object lib/gtest/googlemock/CMakeFiles/gmock_main.dir/__/googletest/src/gtest-all.cc.o"
	cd /home/ygx/src/develop/torchkata/lib/gtest/googlemock && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/gmock_main.dir/__/googletest/src/gtest-all.cc.o -c /home/ygx/src/develop/torchkata/lib/gtest/googletest/src/gtest-all.cc

lib/gtest/googlemock/CMakeFiles/gmock_main.dir/__/googletest/src/gtest-all.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/gmock_main.dir/__/googletest/src/gtest-all.cc.i"
	cd /home/ygx/src/develop/torchkata/lib/gtest/googlemock && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ygx/src/develop/torchkata/lib/gtest/googletest/src/gtest-all.cc > CMakeFiles/gmock_main.dir/__/googletest/src/gtest-all.cc.i

lib/gtest/googlemock/CMakeFiles/gmock_main.dir/__/googletest/src/gtest-all.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/gmock_main.dir/__/googletest/src/gtest-all.cc.s"
	cd /home/ygx/src/develop/torchkata/lib/gtest/googlemock && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ygx/src/develop/torchkata/lib/gtest/googletest/src/gtest-all.cc -o CMakeFiles/gmock_main.dir/__/googletest/src/gtest-all.cc.s

lib/gtest/googlemock/CMakeFiles/gmock_main.dir/src/gmock-all.cc.o: lib/gtest/googlemock/CMakeFiles/gmock_main.dir/flags.make
lib/gtest/googlemock/CMakeFiles/gmock_main.dir/src/gmock-all.cc.o: lib/gtest/googlemock/src/gmock-all.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ygx/src/develop/torchkata/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object lib/gtest/googlemock/CMakeFiles/gmock_main.dir/src/gmock-all.cc.o"
	cd /home/ygx/src/develop/torchkata/lib/gtest/googlemock && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/gmock_main.dir/src/gmock-all.cc.o -c /home/ygx/src/develop/torchkata/lib/gtest/googlemock/src/gmock-all.cc

lib/gtest/googlemock/CMakeFiles/gmock_main.dir/src/gmock-all.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/gmock_main.dir/src/gmock-all.cc.i"
	cd /home/ygx/src/develop/torchkata/lib/gtest/googlemock && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ygx/src/develop/torchkata/lib/gtest/googlemock/src/gmock-all.cc > CMakeFiles/gmock_main.dir/src/gmock-all.cc.i

lib/gtest/googlemock/CMakeFiles/gmock_main.dir/src/gmock-all.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/gmock_main.dir/src/gmock-all.cc.s"
	cd /home/ygx/src/develop/torchkata/lib/gtest/googlemock && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ygx/src/develop/torchkata/lib/gtest/googlemock/src/gmock-all.cc -o CMakeFiles/gmock_main.dir/src/gmock-all.cc.s

lib/gtest/googlemock/CMakeFiles/gmock_main.dir/src/gmock_main.cc.o: lib/gtest/googlemock/CMakeFiles/gmock_main.dir/flags.make
lib/gtest/googlemock/CMakeFiles/gmock_main.dir/src/gmock_main.cc.o: lib/gtest/googlemock/src/gmock_main.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ygx/src/develop/torchkata/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object lib/gtest/googlemock/CMakeFiles/gmock_main.dir/src/gmock_main.cc.o"
	cd /home/ygx/src/develop/torchkata/lib/gtest/googlemock && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/gmock_main.dir/src/gmock_main.cc.o -c /home/ygx/src/develop/torchkata/lib/gtest/googlemock/src/gmock_main.cc

lib/gtest/googlemock/CMakeFiles/gmock_main.dir/src/gmock_main.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/gmock_main.dir/src/gmock_main.cc.i"
	cd /home/ygx/src/develop/torchkata/lib/gtest/googlemock && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ygx/src/develop/torchkata/lib/gtest/googlemock/src/gmock_main.cc > CMakeFiles/gmock_main.dir/src/gmock_main.cc.i

lib/gtest/googlemock/CMakeFiles/gmock_main.dir/src/gmock_main.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/gmock_main.dir/src/gmock_main.cc.s"
	cd /home/ygx/src/develop/torchkata/lib/gtest/googlemock && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ygx/src/develop/torchkata/lib/gtest/googlemock/src/gmock_main.cc -o CMakeFiles/gmock_main.dir/src/gmock_main.cc.s

# Object files for target gmock_main
gmock_main_OBJECTS = \
"CMakeFiles/gmock_main.dir/__/googletest/src/gtest-all.cc.o" \
"CMakeFiles/gmock_main.dir/src/gmock-all.cc.o" \
"CMakeFiles/gmock_main.dir/src/gmock_main.cc.o"

# External object files for target gmock_main
gmock_main_EXTERNAL_OBJECTS =

lib/gtest/googlemock/libgmock_main.a: lib/gtest/googlemock/CMakeFiles/gmock_main.dir/__/googletest/src/gtest-all.cc.o
lib/gtest/googlemock/libgmock_main.a: lib/gtest/googlemock/CMakeFiles/gmock_main.dir/src/gmock-all.cc.o
lib/gtest/googlemock/libgmock_main.a: lib/gtest/googlemock/CMakeFiles/gmock_main.dir/src/gmock_main.cc.o
lib/gtest/googlemock/libgmock_main.a: lib/gtest/googlemock/CMakeFiles/gmock_main.dir/build.make
lib/gtest/googlemock/libgmock_main.a: lib/gtest/googlemock/CMakeFiles/gmock_main.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/ygx/src/develop/torchkata/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Linking CXX static library libgmock_main.a"
	cd /home/ygx/src/develop/torchkata/lib/gtest/googlemock && $(CMAKE_COMMAND) -P CMakeFiles/gmock_main.dir/cmake_clean_target.cmake
	cd /home/ygx/src/develop/torchkata/lib/gtest/googlemock && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/gmock_main.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
lib/gtest/googlemock/CMakeFiles/gmock_main.dir/build: lib/gtest/googlemock/libgmock_main.a

.PHONY : lib/gtest/googlemock/CMakeFiles/gmock_main.dir/build

lib/gtest/googlemock/CMakeFiles/gmock_main.dir/clean:
	cd /home/ygx/src/develop/torchkata/lib/gtest/googlemock && $(CMAKE_COMMAND) -P CMakeFiles/gmock_main.dir/cmake_clean.cmake
.PHONY : lib/gtest/googlemock/CMakeFiles/gmock_main.dir/clean

lib/gtest/googlemock/CMakeFiles/gmock_main.dir/depend:
	cd /home/ygx/src/develop/torchkata && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/ygx/src/develop/torchkata /home/ygx/src/develop/torchkata/lib/gtest/googlemock /home/ygx/src/develop/torchkata /home/ygx/src/develop/torchkata/lib/gtest/googlemock /home/ygx/src/develop/torchkata/lib/gtest/googlemock/CMakeFiles/gmock_main.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : lib/gtest/googlemock/CMakeFiles/gmock_main.dir/depend
