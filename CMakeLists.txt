

cmake_minimum_required(VERSION 3.14.0)

cmake_policy(SET CMP0048 NEW)


project(kinect_mocap_studio LANGUAGES C CXX
    VERSION 1.4)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_STANDARD 99)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(k4a REQUIRED)
find_package(k4abt REQUIRED)
find_package(k4arecord REQUIRED)

# These specific settings tell the loader to search the directory of the
# executable for shared objects. This is done on Linux to emulate the default
# behavior of the Windows loader, which searches for DLLs in the path of the
# executable.
if ("${CMAKE_SYSTEM_NAME}" STREQUAL "Linux")
    set(CMAKE_BUILD_RPATH "\$ORIGIN")
endif()

# If using clang or GCC, set default visibilty to hidden
if ("${CMAKE_C_COMPILER_ID}" STREQUAL "GNU" OR "${CMAKE_C_COMPILER_ID}" STREQUAL "Clang")
    set(CMAKE_CXX_VISIBILITY_PRESET hidden)
    set(CMAKE_C_VISIBILITY_PRESET hidden)
    set(CMAKE_VISIBILITY_INLINES_HIDDEN 1)
endif()

# If using clang or GCC only linked shared libraries if needed
if ("${CMAKE_C_COMPILER_ID}" STREQUAL "GNU" OR "${CMAKE_C_COMPILER_ID}" STREQUAL "Clang")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--as-needed,-rpath-link=${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--as-needed,-rpath-link=${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")
endif()

include(FetchContent)
include(ExternalProject)

FetchContent_Declare(
    json
    URL https://github.com/nlohmann/json/releases/download/v3.11.2/json.tar.xz
)
FetchContent_MakeAvailable(json)

set(EIGEN_BUILD_DIR   ${CMAKE_BINARY_DIR}/eigen)
set(EIGEN_INSTALL_DIR ${CMAKE_SOURCE_DIR}/include/eigen3)
set(EIGEN_INCLUDE_DIR ${EIGEN_INSTALL_DIR})

ExternalProject_Add(
    eigen
    URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
    PREFIX ${EIGEN_BUILD_DIR}
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND
      ${CMAKE_COMMAND} -E copy_directory ${EIGEN_BUILD_DIR}/src/eigen/Eigen ${EIGEN_INCLUDE_DIR}/Eigen 
      && ${CMAKE_COMMAND} -E copy_directory ${EIGEN_BUILD_DIR}/src/eigen/unsupported ${EIGEN_INCLUDE_DIR}/unsupported
    TEST_COMMAND ""
)
include_directories(SYSTEM ${EIGEN_INCLUDE_DIR})

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/sample_helper_includes
                    ${CMAKE_CURRENT_SOURCE_DIR}/glfw/src/include
                    ${CMAKE_CURRENT_SOURCE_DIR}/include
                    ${CMAKE_CURRENT_SOURCE_DIR}/include/eigen
                    ${CMAKE_CURRENT_SOURCE_DIR}/floor_detector)
add_subdirectory(glfw)
add_subdirectory(window_controller_3d)


add_executable(kinect_mocap_studio
		src/kinect_mocap_studio.cc
		floor_detector/FloorDetector.cpp
		floor_detector/PointCloudGenerator.cpp)


# Dependencies of this library
target_link_libraries(kinect_mocap_studio PRIVATE
    k4a
    k4abt
    window_controller_3d::window_controller_3d
    k4arecord
    nlohmann_json::nlohmann_json
    )

