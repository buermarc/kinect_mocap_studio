#include <kinect_mocap_studio/process.hpp>
#include <optional>
#include <iostream>
#include <thread>

#include <k4abt.h>

#include "PointCloudGenerator.h"
#include "FloorDetector.h"

#include <nlohmann/json.hpp>

// We read from the frames queue
//
// Needs to fill the render queue with depth images, and skeleton data

std::optional<Samples::Plane> detect_floor(MeasuredFrame frame, k4a_calibration_t sensor_calibration, Samples::PointCloudGenerator& pointCloudGenerator, Samples::FloorDetector& floorDetector, nlohmann::json& frame_result_json) {
    pointCloudGenerator.Update(frame.depth_image);

    // Get down-sampled cloud points.
    const int downsampleStep = 2;
    const auto& cloudPoints = pointCloudGenerator.GetCloudPoints(downsampleStep);

    // Detect floor plane based on latest visual and inertial observations.
    const size_t minimumFloorPointCount = 1024 / (downsampleStep * downsampleStep);
    const auto& maybeFloorPlane = floorDetector.TryDetectFloorPlane(cloudPoints, frame.imu_sample,
        sensor_calibration, minimumFloorPointCount);

    return maybeFloorPlane;
}

void apply_filter(MeasuredFrame frame) {
}

ProcessedFrame processLogic(MeasuredFrame frame, k4a_calibration_t sensor_calibration, Samples::PointCloudGenerator& pointCloudGenerator, Samples::FloorDetector& floorDetector, nlohmann::json& frame_result_json) {
    // Can we detect the floor
    auto optional_point = detect_floor(frame, sensor_calibration, pointCloudGenerator, floorDetector, frame_result_json);
    apply_filter(frame);

    return ProcessedFrame { {{1.}}, frame.depth_image, optional_point };
}

void processThread(k4a_calibration_t sensor_calibration) {
    Samples::PointCloudGenerator pointCloudGenerator { sensor_calibration };
    Samples::FloorDetector floorDetector;
    MeasuredFrame frame;

    nlohmann::json frame_result_json;

    while (s_isRunning) {
        while (measurement_queue.pop(frame)) {
            std::cout << "Get element from measurement queue" << std::endl;
            ProcessedFrame result = processLogic(frame, sensor_calibration, pointCloudGenerator, floorDetector, frame_result_json);
            std::cout << "Put element on processed queue" << std::endl;
            processed_queue.push(result);

            // Make sure to relase the body frame
            k4abt_frame_release(frame.body_frame);
        }
        std::this_thread::yield();
        // std::cout << "Measurement pop failed." << std::endl;
    }
}
