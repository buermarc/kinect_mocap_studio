#pragma once
#include <future>
#include <tuple>

#include <kinect_mocap_studio/benchmark.hpp>
#include <kinect_mocap_studio/moving_average.hpp>
#include <kinect_mocap_studio/queues.hpp>
#include <kinect_mocap_studio/plotwrap.hpp>

#include "PointCloudGenerator.h"
#include "FloorDetector.h"

#include <nlohmann/json.hpp>

ProcessedFrame processLogic(MeasuredFrame frame, k4a_calibration_t sensor_calibration, Samples::FloorDetector& floorDetector, nlohmann::json& frame_result_json, MovingAverage& moving_average);
void processThread(k4a_calibration_t sensor_calibration, std::promise<std::tuple<nlohmann::json, PlotWrap<double>>>, Benchmark& bench);

std::optional<Samples::Plane> detect_floor(MeasuredFrame frame, k4a_calibration_t sensor_calibration, Samples::PointCloudGenerator& pointCloudGenerator, Samples::FloorDetector& floorDetector, nlohmann::json& frame_result_json, MovingAverage& moving_average);
void apply_filter(MeasuredFrame frame);
