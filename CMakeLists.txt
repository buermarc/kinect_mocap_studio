

cmake_minimum_required(VERSION 3.14.0)

# cmake_policy(SET CMP0048 NEW)


project(kinect_mocap_studio LANGUAGES C CXX VERSION 1.4)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
# set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(BENCH_MEASUREMENT 0)
set(BENCH_PROCESS 0)
set(BENCH_VIZ 0)

find_package(k4a REQUIRED)
find_package(k4abt REQUIRED)
find_package(k4arecord REQUIRED)
find_package(PythonLibs)
find_package(alglib)
include_directories(${PYTHON_INCLUDE_DIRS})

# These specific settings tell the loader to search the directory of the
# executable for shared objects. This is done on Linux to emulate the default
# behavior of the Windows loader, which searches for DLLs in the path of the
# executable.
# if ("${CMAKE_SYSTEM_NAME}" STREQUAL "Linux")
#     set(CMAKE_BUILD_RPATH "\$ORIGIN")
# endif()

# If using clang or GCC, set default visibilty to hidden
# if ("${CMAKE_C_COMPILER_ID}" STREQUAL "GNU" OR "${CMAKE_C_COMPILER_ID}" STREQUAL "Clang")
#     set(CMAKE_CXX_VISIBILITY_PRESET hidden)
#     set(CMAKE_C_VISIBILITY_PRESET hidden)
#     set(CMAKE_VISIBILITY_INLINES_HIDDEN 1)
# endif()
#
# # If using clang or GCC only linked shared libraries if needed
# if ("${CMAKE_C_COMPILER_ID}" STREQUAL "GNU" OR "${CMAKE_C_COMPILER_ID}" STREQUAL "Clang")
#     set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--as-needed,-rpath-link=${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")
#     set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--as-needed,-rpath-link=${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")
# endif()

include(FetchContent)
include(ExternalProject)

FetchContent_Declare(
    json
    URL https://github.com/nlohmann/json/releases/download/v3.11.2/json.tar.xz
)
FetchContent_MakeAvailable(json)

### Boost (for lockfree sharde queue)
set(BOOST_INCLUDE_LIBRARIES archive serialization lockfree)
set(BOOST_ENABLE_CMAKE ON)

include(FetchContent)
FetchContent_Declare(
  Boost
  URL https://github.com/boostorg/boost/releases/download/boost-1.83.0/boost-1.83.0.tar.xz
)
FetchContent_MakeAvailable(Boost)


include_directories(${CMAKE_CURRENT_SOURCE_DIR}/sample_helper_includes
                    ${CMAKE_CURRENT_SOURCE_DIR}/glfw/src/include
                    ${CMAKE_CURRENT_SOURCE_DIR}/include
                    ${CMAKE_CURRENT_SOURCE_DIR}/floor_detector
                    ${CMAKE_CURRENT_SOURCE_DIR}/openGLPlotLive/src)

FetchContent_Declare(
  filter
  GIT_REPOSITORY https://github.com/buermarc/master-code.git
  GIT_TAG HEAD
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
FetchContent_MakeAvailable(filter)

FetchContent_Declare(
  iir1
  GIT_REPOSITORY https://github.com/berndporr/iir1
  GIT_TAG HEAD
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
FetchContent_MakeAvailable(iir1)

FetchContent_Declare(
  cnpy
  GIT_REPOSITORY https://github.com/rogersce/cnpy.git
  GIT_TAG HEAD
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
FetchContent_MakeAvailable(cnpy)

set(EIGEN_BUILD_DIR   ${CMAKE_BINARY_DIR}/eigen)
set(EIGEN_INSTALL_DIR ${CMAKE_SOURCE_DIR}/include/eigen3)
set(EIGEN_INCLUDE_DIR ${EIGEN_INSTALL_DIR})
include_directories(SYSTEM ${EIGEN_INCLUDE_DIR})

add_subdirectory(glfw)
add_subdirectory(window_controller_3d)
add_subdirectory(openGLPlotLive)

#find_package(FREETYPE REQUIRED)
include_directories(${FREETYPE_INCLUDE_DIR_ft2build})

add_executable(kinect_mocap_studio
		src/kinect_mocap_studio.cpp
		src/utils.cpp
		src/cli.cpp
		src/process.cpp
		src/visualize.cpp
		src/plotting.cpp
		src/moving_average.cpp
		src/benchmark.cpp
		floor_detector/FloorDetector.cpp
		floor_detector/PointCloudGenerator.cpp)

# Dependencies of this library
target_link_libraries(kinect_mocap_studio
    Boost::lockfree
    filter
    k4a
    k4abt
    window_controller_3d::window_controller_3d
    k4arecord
    nlohmann_json::nlohmann_json
    openGLPlotLive
    freetype
    ${PYTHON_LIBRARIES}
)
add_executable(evaluate src/evaluate.cpp)
target_link_libraries(evaluate
    Boost::lockfree
    filter
    k4a
    k4abt
    window_controller_3d::window_controller_3d
    k4arecord
    nlohmann_json::nlohmann_json
    openGLPlotLive
    freetype
    alglib
    iir::iir
    cnpy
    ${PYTHON_LIBRARIES}
)

add_executable(baseline src/baseline.cpp)
target_link_libraries(baseline
    k4a
    k4abt
    k4arecord
    nlohmann_json::nlohmann_json
    filter
)


add_custom_command(TARGET kinect_mocap_studio PRE_BUILD COMMAND ${CMAKE_COMMAND} -E copy_directory ${PROJECT_SOURCE_DIR}/../openGLPlotLive/Shaders $<TARGET_FILE_DIR:kinect_mocap_studio>/Shaders)
add_custom_command(TARGET kinect_mocap_studio PRE_BUILD COMMAND ${CMAKE_COMMAND} -E copy_directory ${PROJECT_SOURCE_DIR}/../openGLPlotLive/textures $<TARGET_FILE_DIR:kinect_mocap_studio>/textures)
