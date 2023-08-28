# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

add_library(floor_detector STATIC
           FloorDetector.cpp
           PointCloudGenerator.cpp)

target_include_directories(floor_detector PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Dependencies of this library
target_link_libraries(floor_detector PRIVATE 
    k4a
    )

add_library(floor_detector::floor_detector ALIAS floor_detector)
