#include <kinect_mocap_studio/process.hpp>

#include <k4abt.h>

#include "PointCloudGenerator.h"
#include "FloorDetector.h"

#include <nlohmann/json.hpp>

// We read from the frames queue
//
// Needs to fill the render queue with depth images, and skeleton data

void detect_floor(MeasuredFrame frame, k4a_calibration_t sensor_calibration, Samples::PointCloudGenerator& pointCloudGenerator, Samples::FloorDetector& floorDetector, nlohmann::json& frame_result_json) {
    pointCloudGenerator.Update(frame.depth_image);

    // Get down-sampled cloud points.
    const int downsampleStep = 2;
    const auto& cloudPoints = pointCloudGenerator.GetCloudPoints(downsampleStep);

    // Detect floor plane based on latest visual and inertial observations.
    const size_t minimumFloorPointCount = 1024 / (downsampleStep * downsampleStep);
    const auto& maybeFloorPlane = floorDetector.TryDetectFloorPlane(cloudPoints, frame.imu_sample,
        sensor_calibration, minimumFloorPointCount);

    // Visualize point cloud.
    // window3d.UpdatePointClouds(depth_image);

    // Visualize the floor plane.
    nlohmann::json floor_result_json;
    if (maybeFloorPlane.has_value()) {
        // For visualization purposes, make floor origin the projection of a point 1.5m in front of the camera.
        Samples::Vector cameraOrigin = { 0, 0, 0 };
        Samples::Vector cameraForward = { 0, 0, 1 };

        auto p = maybeFloorPlane->ProjectPoint(cameraOrigin)
            + maybeFloorPlane->ProjectVector(cameraForward) * 1.5f;

        auto n = maybeFloorPlane->Normal;

        // window3d.SetFloorRendering(true, p.X, p.Y, p.Z, n.X, n.Y, n.Z);
        floor_result_json["point"].push_back(
            { p.X * 1000.f, p.Y * 1000.f, p.Z * 1000.f });
        floor_result_json["normal"].push_back({ n.X, n.Y, n.Z });
        floor_result_json["valid"] = true;
    } else {
        // window3d.SetFloorRendering(false, 0, 0, 0);
        floor_result_json["point"].push_back({ 0., 0., 0. });
        floor_result_json["normal"].push_back({ 0., 0., 0. });
        floor_result_json["valid"] = false;
    }
    frame_result_json["floor"].push_back(floor_result_json);
}

void apply_filter(MeasuredFrame frame) {
}

ProcessedFrame processLogic(MeasuredFrame frame, k4a_calibration_t sensor_calibration, Samples::PointCloudGenerator& pointCloudGenerator, Samples::FloorDetector& floorDetector, nlohmann::json& frame_result_json) {
    // Can we detect the floor
    detect_floor(frame, sensor_calibration, pointCloudGenerator, floorDetector, frame_result_json);
    apply_filter(frame);

    k4abt_frame_release(frame.body_frame);
    // TODO: need to release depth image in the render pipeline, beacuse we need it there
    // k4a_image_release(frame.depth_image);
    return ProcessedFrame { {{1.}}, frame.depth_image};
}

void processThead(k4a_calibration_t sensor_calibration) {
    Samples::PointCloudGenerator pointCloudGenerator { sensor_calibration };
    Samples::FloorDetector floorDetector;
    MeasuredFrame frame;

    nlohmann::json frame_result_json;

    while (s_isRunning) {
        while (measurement_queue.pop(frame)) {
            ProcessedFrame result = processLogic(frame, sensor_calibration, pointCloudGenerator, floorDetector, frame_result_json);
            processed_queue.push(result);
        }
    }
}
