# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

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
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /tmp/34789ce5-f372-4d30-9de7-22d067fa13e3

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /tmp/34789ce5-f372-4d30-9de7-22d067fa13e3/build/cmake-build-ascend-x86_64-linux-release

# Include any dependencies generated for this target.
include src/CMakeFiles/Resnet18_Mxbase.dir/depend.make

# Include the progress variables for this target.
include src/CMakeFiles/Resnet18_Mxbase.dir/progress.make

# Include the compile flags for this target's objects.
include src/CMakeFiles/Resnet18_Mxbase.dir/flags.make

src/CMakeFiles/Resnet18_Mxbase.dir/main.cpp.o: src/CMakeFiles/Resnet18_Mxbase.dir/flags.make
src/CMakeFiles/Resnet18_Mxbase.dir/main.cpp.o: ../../src/main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/tmp/34789ce5-f372-4d30-9de7-22d067fa13e3/build/cmake-build-ascend-x86_64-linux-release/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object src/CMakeFiles/Resnet18_Mxbase.dir/main.cpp.o"
	cd /tmp/34789ce5-f372-4d30-9de7-22d067fa13e3/build/cmake-build-ascend-x86_64-linux-release/src && /usr/bin/g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/Resnet18_Mxbase.dir/main.cpp.o -c /tmp/34789ce5-f372-4d30-9de7-22d067fa13e3/src/main.cpp

src/CMakeFiles/Resnet18_Mxbase.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Resnet18_Mxbase.dir/main.cpp.i"
	cd /tmp/34789ce5-f372-4d30-9de7-22d067fa13e3/build/cmake-build-ascend-x86_64-linux-release/src && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /tmp/34789ce5-f372-4d30-9de7-22d067fa13e3/src/main.cpp > CMakeFiles/Resnet18_Mxbase.dir/main.cpp.i

src/CMakeFiles/Resnet18_Mxbase.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Resnet18_Mxbase.dir/main.cpp.s"
	cd /tmp/34789ce5-f372-4d30-9de7-22d067fa13e3/build/cmake-build-ascend-x86_64-linux-release/src && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /tmp/34789ce5-f372-4d30-9de7-22d067fa13e3/src/main.cpp -o CMakeFiles/Resnet18_Mxbase.dir/main.cpp.s

src/CMakeFiles/Resnet18_Mxbase.dir/main.cpp.o.requires:

.PHONY : src/CMakeFiles/Resnet18_Mxbase.dir/main.cpp.o.requires

src/CMakeFiles/Resnet18_Mxbase.dir/main.cpp.o.provides: src/CMakeFiles/Resnet18_Mxbase.dir/main.cpp.o.requires
	$(MAKE) -f src/CMakeFiles/Resnet18_Mxbase.dir/build.make src/CMakeFiles/Resnet18_Mxbase.dir/main.cpp.o.provides.build
.PHONY : src/CMakeFiles/Resnet18_Mxbase.dir/main.cpp.o.provides

src/CMakeFiles/Resnet18_Mxbase.dir/main.cpp.o.provides.build: src/CMakeFiles/Resnet18_Mxbase.dir/main.cpp.o


src/CMakeFiles/Resnet18_Mxbase.dir/Resnet18Classify.cpp.o: src/CMakeFiles/Resnet18_Mxbase.dir/flags.make
src/CMakeFiles/Resnet18_Mxbase.dir/Resnet18Classify.cpp.o: ../../src/Resnet18Classify.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/tmp/34789ce5-f372-4d30-9de7-22d067fa13e3/build/cmake-build-ascend-x86_64-linux-release/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object src/CMakeFiles/Resnet18_Mxbase.dir/Resnet18Classify.cpp.o"
	cd /tmp/34789ce5-f372-4d30-9de7-22d067fa13e3/build/cmake-build-ascend-x86_64-linux-release/src && /usr/bin/g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/Resnet18_Mxbase.dir/Resnet18Classify.cpp.o -c /tmp/34789ce5-f372-4d30-9de7-22d067fa13e3/src/Resnet18Classify.cpp

src/CMakeFiles/Resnet18_Mxbase.dir/Resnet18Classify.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Resnet18_Mxbase.dir/Resnet18Classify.cpp.i"
	cd /tmp/34789ce5-f372-4d30-9de7-22d067fa13e3/build/cmake-build-ascend-x86_64-linux-release/src && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /tmp/34789ce5-f372-4d30-9de7-22d067fa13e3/src/Resnet18Classify.cpp > CMakeFiles/Resnet18_Mxbase.dir/Resnet18Classify.cpp.i

src/CMakeFiles/Resnet18_Mxbase.dir/Resnet18Classify.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Resnet18_Mxbase.dir/Resnet18Classify.cpp.s"
	cd /tmp/34789ce5-f372-4d30-9de7-22d067fa13e3/build/cmake-build-ascend-x86_64-linux-release/src && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /tmp/34789ce5-f372-4d30-9de7-22d067fa13e3/src/Resnet18Classify.cpp -o CMakeFiles/Resnet18_Mxbase.dir/Resnet18Classify.cpp.s

src/CMakeFiles/Resnet18_Mxbase.dir/Resnet18Classify.cpp.o.requires:

.PHONY : src/CMakeFiles/Resnet18_Mxbase.dir/Resnet18Classify.cpp.o.requires

src/CMakeFiles/Resnet18_Mxbase.dir/Resnet18Classify.cpp.o.provides: src/CMakeFiles/Resnet18_Mxbase.dir/Resnet18Classify.cpp.o.requires
	$(MAKE) -f src/CMakeFiles/Resnet18_Mxbase.dir/build.make src/CMakeFiles/Resnet18_Mxbase.dir/Resnet18Classify.cpp.o.provides.build
.PHONY : src/CMakeFiles/Resnet18_Mxbase.dir/Resnet18Classify.cpp.o.provides

src/CMakeFiles/Resnet18_Mxbase.dir/Resnet18Classify.cpp.o.provides.build: src/CMakeFiles/Resnet18_Mxbase.dir/Resnet18Classify.cpp.o


# Object files for target Resnet18_Mxbase
Resnet18_Mxbase_OBJECTS = \
"CMakeFiles/Resnet18_Mxbase.dir/main.cpp.o" \
"CMakeFiles/Resnet18_Mxbase.dir/Resnet18Classify.cpp.o"

# External object files for target Resnet18_Mxbase
Resnet18_Mxbase_EXTERNAL_OBJECTS =

../../out/Resnet18_Mxbase: src/CMakeFiles/Resnet18_Mxbase.dir/main.cpp.o
../../out/Resnet18_Mxbase: src/CMakeFiles/Resnet18_Mxbase.dir/Resnet18Classify.cpp.o
../../out/Resnet18_Mxbase: src/CMakeFiles/Resnet18_Mxbase.dir/build.make
../../out/Resnet18_Mxbase: src/CMakeFiles/Resnet18_Mxbase.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/tmp/34789ce5-f372-4d30-9de7-22d067fa13e3/build/cmake-build-ascend-x86_64-linux-release/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable ../../../out/Resnet18_Mxbase"
	cd /tmp/34789ce5-f372-4d30-9de7-22d067fa13e3/build/cmake-build-ascend-x86_64-linux-release/src && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/Resnet18_Mxbase.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/CMakeFiles/Resnet18_Mxbase.dir/build: ../../out/Resnet18_Mxbase

.PHONY : src/CMakeFiles/Resnet18_Mxbase.dir/build

src/CMakeFiles/Resnet18_Mxbase.dir/requires: src/CMakeFiles/Resnet18_Mxbase.dir/main.cpp.o.requires
src/CMakeFiles/Resnet18_Mxbase.dir/requires: src/CMakeFiles/Resnet18_Mxbase.dir/Resnet18Classify.cpp.o.requires

.PHONY : src/CMakeFiles/Resnet18_Mxbase.dir/requires

src/CMakeFiles/Resnet18_Mxbase.dir/clean:
	cd /tmp/34789ce5-f372-4d30-9de7-22d067fa13e3/build/cmake-build-ascend-x86_64-linux-release/src && $(CMAKE_COMMAND) -P CMakeFiles/Resnet18_Mxbase.dir/cmake_clean.cmake
.PHONY : src/CMakeFiles/Resnet18_Mxbase.dir/clean

src/CMakeFiles/Resnet18_Mxbase.dir/depend:
	cd /tmp/34789ce5-f372-4d30-9de7-22d067fa13e3/build/cmake-build-ascend-x86_64-linux-release && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /tmp/34789ce5-f372-4d30-9de7-22d067fa13e3 /tmp/34789ce5-f372-4d30-9de7-22d067fa13e3/src /tmp/34789ce5-f372-4d30-9de7-22d067fa13e3/build/cmake-build-ascend-x86_64-linux-release /tmp/34789ce5-f372-4d30-9de7-22d067fa13e3/build/cmake-build-ascend-x86_64-linux-release/src /tmp/34789ce5-f372-4d30-9de7-22d067fa13e3/build/cmake-build-ascend-x86_64-linux-release/src/CMakeFiles/Resnet18_Mxbase.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/CMakeFiles/Resnet18_Mxbase.dir/depend

