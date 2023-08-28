include(CMakeFindDependencyMacro)

# Define kinect_mocap_studio targets
add_library(kinect_mocap_studio SHARED IMPORTED)
set_property(TARGET kinect_mocap_studio PROPERTY IMPORTED_LOCATION "${CMAKE_INSTALL_PREFIX}/lib/libkinect_mocap_studio.so")

# Set kinect_mocap_studio include directories
# set(kinect_mocap_studio_INCLUDE_DIRS "${CMAKE_INSTALL_PREFIX}/include")
set(kinect_mocap_studio_INCLUDE_DIRS "${CMAKE_INSTALL_PREFIX}/src" "${CMAKE_INSTALL_PREFIX}/floor_detector" "${CMAKE_INSTALL_PREFIX}/sample_helper_includes" "${CMAKE_INSTALL_PREFIX}/window_controller_3d" "${CMAKE_INSTALL_PREFIX}/glfw" "${CMAKE_INSTALL_PREFIX}/glfw/src/include")


# Add kinect_mocap_studio to CMake's package registry
set(kinect_mocap_studio_FOUND TRUE)
set(kinect_mocap_studio_VERSION ${PROJECT_VERSION})

# Define kinect_mocap_studio dependency
# find_dependency(OtherDependency REQUIRED)

# Provide kinect_mocap_studio target to the user
add_library(kinect_mocap_studio::kinect_mocap_studio ALIAS kinect_mocap_studio)

# Set the include directory for external projects
set(kinect_mocap_studio_INCLUDE_DIRS "${CMAKE_INSTALL_PREFIX}/include" CACHE INTERNAL "")
