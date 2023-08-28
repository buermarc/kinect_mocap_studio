cmake_minimum_required(VERSION 3.10)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

project(gui C CXX)

set(INC_DIR inc)
set(SRC_DIR src)

set(HEADER_FILES gui.h)
set(SOURCE_FILES gui.cpp)

# target_include_directories(libgui PUBLIC .)
add_compile_options(-g)

find_package(PkgConfig REQUIRED)
# pkg_check_modules(GTK REQUIRED gtk4)
pkg_check_modules(GTKMM REQUIRED gtkmm-4.0)
# set(libgui ${GTK_LIBRARIES})
# message(GTKMM_INC_DIRS=${PkgConfig})
# message(GTKMM_INC_DIRS="${GTKMM_INCLUDE_DIRS}")
# message(GTK_INC_DIRS="${GTK_INCLUDE_DIRS}")
# message(GTK_INC_DIRS="${PROJECT_TARGET}")

add_library(libgui ${SRC_DIR}/${SOURCE_FILES} ${INC_DIR}/${HEADER_FILES})
target_include_directories(libgui PRIVATE ${INC_DIR} ${GTKMM_INCLUDE_DIRS})

# add_library(kinect_mocap_studio STATIC IMPORTED)
# set_target_properties(kinect_mocap_studio PROPERTIES IMPORTED_LOCATION ../kinect_mocap_studio_clean/libkinect_mocap_studio.a)
set(CMAKE_PREFIX_PATH ../kinect_mocap_studio_clean/package ../kinect_mocap_studio_clean/package/window_controller_3d ../kinect_mocap_studio_clean/package/glfw/src/src ../kinect_mocap_studio_clean/package/floor_detector)
# set(CMAKE_PREFIX_PATH /usr/local/lib)
# find_library(KINECT_LIB libkinect_mocap_studio.a)
find_package(kinect_mocap_studio REQUIRED)
find_package(k4a REQUIRED)
find_package(k4abt REQUIRED)
find_package(k4arecord REQUIRED)
find_library(WINDOW_3D_LIB libwindow_controller_3d.a REQUIRED)
find_library(FLOOR_DET libfloor_detector.a REQUIRED)
# find_library(GLFW_LIB libglfw3.a REQUIRED)

target_include_directories(libgui PRIVATE ../kinect_mocap_studio_clean/src ../kinect_mocap_studio_clean/floor_detector ../kinect_mocap_studio_clean/sample_helper_includes ../kinect_mocap_studio_clean/window_controller_3d ../kinect_mocap_studio_clean/glfw ../kinect_mocap_studio_clean/glfw/src/include)
# target_include_directories(libgui PRIVATE ../kinect_mocap_studio_clean/package)
# target_link_libraries(libgui k4a k4abt k4arecord ${GLFW_LIB} ${WINDOW_3D_LIB} kinect_mocap_studio::kinect_mocap_studio)
target_link_libraries(libgui k4a k4abt k4arecord ${GLFW_LIB} ${WINDOW_3D_LIB} ${FLOOR_DET} kinect_mocap_studio::kinect_mocap_studio)


# include_directories(${GTK_INCLUDE_DIRS} ${GTKMM_INCLUDE_DIRS} ${INC_DIR})
# link_directories(${GTK_LIBRARY_DIRS} ${GTKMM_LIBRARY_DIRS} ${INC_DIR})
# include_directories(${INC_DIR})
# link_directories(${GTK_LIBRARY_DIRS} ${GTKMM_LIBRARY_DIRS} ${INC_DIR})

add_executable(${PROJECT_NAME} ${SRC_DIR}/main.cpp)
target_link_libraries(${PROJECT_NAME} PRIVATE ${GTKMM_LIBRARIES} libgui)

# target_compile_options(${PROJECT_NAME} PRIVATE ${GTK_CFLAGS_OTHER})


target_include_directories(${PROJECT_NAME} PRIVATE ${GTKMM_INCLUDE_DIRS} ${INC_DIR})
target_link_directories(${PROJECT_NAME} PRIVATE ${GTKMM_LIBRARY_DIRS} ${INC_DIR})
# message(GTK_CFLAGS_OTHER="${GTK_CFLAGS_OTHER}")
# message(GTKMM_CFLAGS_OTHER="${GTKMM_CFLAGS_OTHER}")


