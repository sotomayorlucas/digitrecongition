# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

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
CMAKE_SOURCE_DIR = /home/dete/Desktop/Gits/tp2

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/dete/Desktop/Gits/tp2

# Include any dependencies generated for this target.
include CMakeFiles/tp2_pybind.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/tp2_pybind.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/tp2_pybind.dir/flags.make

CMakeFiles/tp2_pybind.dir/src/main.cpp.o: CMakeFiles/tp2_pybind.dir/flags.make
CMakeFiles/tp2_pybind.dir/src/main.cpp.o: src/main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/dete/Desktop/Gits/tp2/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/tp2_pybind.dir/src/main.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/tp2_pybind.dir/src/main.cpp.o -c /home/dete/Desktop/Gits/tp2/src/main.cpp

CMakeFiles/tp2_pybind.dir/src/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/tp2_pybind.dir/src/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/dete/Desktop/Gits/tp2/src/main.cpp > CMakeFiles/tp2_pybind.dir/src/main.cpp.i

CMakeFiles/tp2_pybind.dir/src/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/tp2_pybind.dir/src/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/dete/Desktop/Gits/tp2/src/main.cpp -o CMakeFiles/tp2_pybind.dir/src/main.cpp.s

CMakeFiles/tp2_pybind.dir/src/funciones.cpp.o: CMakeFiles/tp2_pybind.dir/flags.make
CMakeFiles/tp2_pybind.dir/src/funciones.cpp.o: src/funciones.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/dete/Desktop/Gits/tp2/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/tp2_pybind.dir/src/funciones.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/tp2_pybind.dir/src/funciones.cpp.o -c /home/dete/Desktop/Gits/tp2/src/funciones.cpp

CMakeFiles/tp2_pybind.dir/src/funciones.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/tp2_pybind.dir/src/funciones.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/dete/Desktop/Gits/tp2/src/funciones.cpp > CMakeFiles/tp2_pybind.dir/src/funciones.cpp.i

CMakeFiles/tp2_pybind.dir/src/funciones.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/tp2_pybind.dir/src/funciones.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/dete/Desktop/Gits/tp2/src/funciones.cpp -o CMakeFiles/tp2_pybind.dir/src/funciones.cpp.s

# Object files for target tp2_pybind
tp2_pybind_OBJECTS = \
"CMakeFiles/tp2_pybind.dir/src/main.cpp.o" \
"CMakeFiles/tp2_pybind.dir/src/funciones.cpp.o"

# External object files for target tp2_pybind
tp2_pybind_EXTERNAL_OBJECTS =

tp2_pybind: CMakeFiles/tp2_pybind.dir/src/main.cpp.o
tp2_pybind: CMakeFiles/tp2_pybind.dir/src/funciones.cpp.o
tp2_pybind: CMakeFiles/tp2_pybind.dir/build.make
tp2_pybind: CMakeFiles/tp2_pybind.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/dete/Desktop/Gits/tp2/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable tp2_pybind"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/tp2_pybind.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/tp2_pybind.dir/build: tp2_pybind

.PHONY : CMakeFiles/tp2_pybind.dir/build

CMakeFiles/tp2_pybind.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/tp2_pybind.dir/cmake_clean.cmake
.PHONY : CMakeFiles/tp2_pybind.dir/clean

CMakeFiles/tp2_pybind.dir/depend:
	cd /home/dete/Desktop/Gits/tp2 && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/dete/Desktop/Gits/tp2 /home/dete/Desktop/Gits/tp2 /home/dete/Desktop/Gits/tp2 /home/dete/Desktop/Gits/tp2 /home/dete/Desktop/Gits/tp2/CMakeFiles/tp2_pybind.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/tp2_pybind.dir/depend

