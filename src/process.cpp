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

std::optional<Samples::Plane> detect_floor(MeasuredFrame frame, k4a_calibration_t sensor_calibration, Samples::FloorDetector& floorDetector, nlohmann::json& frame_result_json) {
    // Get down-sampled cloud points.
    const int downsampleStep = 2;
    // Detect floor plane based on latest visual and inertial observations.
    const size_t minimumFloorPointCount = 1024 / (downsampleStep * downsampleStep);

    // auto pointCloud = Samples::ConvertPointCloud(frame.cloudPoints);
    const auto& maybeFloorPlane = floorDetector.TryDetectFloorPlane(frame.cloudPoints, frame.imu_sample,
        sensor_calibration, minimumFloorPointCount);

    return maybeFloorPlane;
}

void apply_filter(MeasuredFrame frame) {
}

ProcessedFrame processLogic(MeasuredFrame frame, k4a_calibration_t sensor_calibration, Samples::FloorDetector& floorDetector, nlohmann::json& frame_result_json) {
    // Can we detect the floor
    auto optional_point = detect_floor(frame, sensor_calibration, floorDetector, frame_result_json);
    apply_filter(frame);

    return ProcessedFrame { frame.cloudPoints, frame.joints, optional_point };
}

void processThread(k4a_calibration_t sensor_calibration) {
    Samples::PointCloudGenerator pointCloudGenerator { sensor_calibration };
    Samples::FloorDetector floorDetector;
    MeasuredFrame frame;

    nlohmann::json frame_result_json;

    while (s_isRunning) {
        bool retrieved = measurement_queue.Consume(frame);
        if (retrieved) {
            ProcessedFrame result = processLogic(frame, sensor_calibration, floorDetector, frame_result_json);
            processed_queue.Produce(std::move(result));

            // Make sure to relase the body frame
        } else {
            std::this_thread::yield();
        }
    }
}
