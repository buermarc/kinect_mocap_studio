#pragma once
#include <future>

#include <kinect_mocap_studio/moving_average.hpp>
#include <kinect_mocap_studio/queues.hpp>

#include "PointCloudGenerator.h"
#include "FloorDetector.h"

#include <nlohmann/json.hpp>

ProcessedFrame processLogic(MeasuredFrame frame, k4a_calibration_t sensor_calibration, Samples::FloorDetector& floorDetector, nlohmann::json& frame_result_json, MovingAverage& moving_average);
void processThread(k4a_calibration_t sensor_calibration, std::promise<nlohmann::json>);

std::optional<Samples::Plane> detect_floor(MeasuredFrame frame, k4a_calibration_t sensor_calibration, Samples::PointCloudGenerator& pointCloudGenerator, Samples::FloorDetector& floorDetector, nlohmann::json& frame_result_json, MovingAverage& moving_average);
void apply_filter(MeasuredFrame frame);
