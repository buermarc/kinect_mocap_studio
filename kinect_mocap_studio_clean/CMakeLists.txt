

cmake_minimum_required(VERSION 3.9.0)

cmake_policy(SET CMP0048 NEW)


project(kinect_mocap_studio LANGUAGES C CXX
    VERSION 1.4)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_STANDARD 99)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})

add_compile_options(-g)

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



add_library(kinect_mocap_studio SHARED)

# Add the source files to the library target
target_sources(kinect_mocap_studio PRIVATE
               src/kinect_mocap_studio.cc
               floor_detector/FloorDetector.cpp
               floor_detector/PointCloudGenerator.cpp)

include_directories(kinect_mocap_studio PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/sample_helper_includes>
    $<INSTALL_INTERFACE:sample_helper_includes>
)
target_include_directories(kinect_mocap_studio PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/glfw/src/include>
    $<INSTALL_INTERFACE:/glfw/src/include>
)
target_include_directories(kinect_mocap_studio PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/floor_detector>
    $<INSTALL_INTERFACE:floor_detector>
)

add_subdirectory(glfw)
add_subdirectory(window_controller_3d)
add_subdirectory(floor_detector)

# target_include_directories(kinect_mocap_studio PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/sample_helper_includes
#                     ${CMAKE_CURRENT_SOURCE_DIR}/glfw/src/include
#                     ${CMAKE_CURRENT_SOURCE_DIR}/floor_detector)


# Dependencies of this library
target_link_libraries(kinect_mocap_studio PRIVATE 
                      k4a
                      k4abt
                      window_controller_3d::window_controller_3d
                      floor_detector::floor_detector
                      k4arecord
                      )

# Set the library version (optional)
set_target_properties(kinect_mocap_studio PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION 1
)

# Install the library and headers
install(TARGETS kinect_mocap_studio
    EXPORT kinect_mocap_studioTargets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include
)

install(DIRECTORY include/ DESTINATION include)

# Generate and install the CMake package configuration files
install(EXPORT kinect_mocap_studioTargets
    FILE kinect_mocap_studioTargets.cmake
    NAMESPACE kinect_mocap_studio::
    DESTINATION lib/cmake/kinect_mocap_studio
)

# Include CMakePackageConfigHelpers to generate version file
include(CMakePackageConfigHelpers)

# Generate and install the package configuration and version files
write_basic_package_version_file(
    "kinect_mocap_studioConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
)

# Generate and install the package configuration file
configure_package_config_file(
    "kinect_mocap_studioConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/kinect_mocap_studioConfig.cmake"
    INSTALL_DESTINATION lib/cmake/kinect_mocap_studio
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/kinect_mocap_studioConfigVersion.cmake
    DESTINATION lib/cmake/kinect_mocap_studio
)
