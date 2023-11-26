#include <kinect_mocap_studio/process.hpp>
#include <optional>
#include <iostream>
#include <thread>

#include <k4abt.h>

#include "PointCloudGenerator.h"
#include "FloorDetector.h"

#include <nlohmann/json.hpp>
#include <filter/SkeletonFilter.hpp>

SkeletonFilterBuilder<double> builder(32, 2.0);

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

void apply_filter(MeasuredFrame& frame, std::vector<SkeletonFilter<double>>& filters) {
    for (int i = 0; i < frame.joints.size(); ++i)
    {
        if (filters.empty() or filters.size() <= frame.joints.size()) {
            filters.push_back(builder.build());
        }

        auto& filter = filters.at(i);
        auto [filtered_positions, _ ] = filter.step(frame.joints.at(i), frame.timestamp);
        frame.joints.at(i) = std::move(filtered_positions);
    }

}

ProcessedFrame processLogic(
    MeasuredFrame frame,
    k4a_calibration_t sensor_calibration,
    Samples::FloorDetector& floorDetector,
    std::vector<SkeletonFilter<double>>& filters,
    nlohmann::json& frame_result_json
) {
    // Can we detect the floor
    auto optional_point = detect_floor(frame, sensor_calibration, floorDetector, frame_result_json);
    // Mutates joints
    apply_filter(frame, filters);

    return ProcessedFrame { frame.cloudPoints, frame.joints, optional_point };
}

void processThread(k4a_calibration_t sensor_calibration) {
    Samples::PointCloudGenerator pointCloudGenerator { sensor_calibration };
    Samples::FloorDetector floorDetector;
    MeasuredFrame frame;
    std::vector<SkeletonFilter<double>> filters;

    nlohmann::json frame_result_json;

    while (s_isRunning) {
        bool retrieved = measurement_queue.Consume(frame);
        if (retrieved) {
            ProcessedFrame result = processLogic(frame, sensor_calibration, floorDetector, filters, frame_result_json);
            processed_queue.Produce(std::move(result));

            // Make sure to relase the body frame
        } else {
            std::this_thread::yield();
        }
    }
}
