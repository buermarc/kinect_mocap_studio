#pragma once
#include <kinect_mocap_studio/queues.hpp>

#include "PointCloudGenerator.h"
#include "FloorDetector.h"

#include <nlohmann/json.hpp>

ProcessedFrame processLogic(MeasuredFrame frame, k4a_calibration_t sensor_calibration, Samples::PointCloudGenerator& pointCloudGenerator, Samples::FloorDetector& floorDetector, nlohmann::json& frame_result_json);
void processThead(k4a_calibration_t sensor_calibration);

void detect_floor(MeasuredFrame frame, k4a_calibration_t sensor_calibration, Samples::PointCloudGenerator& pointCloudGenerator, Samples::FloorDetector& floorDetector, nlohmann::json& frame_result_json);
void apply_filter(MeasuredFrame frame);
