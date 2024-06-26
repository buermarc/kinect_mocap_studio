#pragma once
#include <future>
#include <tuple>

#include <kinect_mocap_studio/benchmark.hpp>
#include <kinect_mocap_studio/moving_average.hpp>
#include <kinect_mocap_studio/queues.hpp>
#include <kinect_mocap_studio/plotwrap.hpp>
#include <filter/AbstractSkeletonFilter.hpp>

#include "PointCloudGenerator.h"
#include "FloorDetector.h"

#include <nlohmann/json.hpp>

typedef std::shared_ptr<AbstractSkeletonFilter<double>> CurrentFilterType;

ProcessedFrame processLogic(MeasuredFrame frame, k4a_calibration_t sensor_calibration, Samples::FloorDetector& floorDetector, nlohmann::json& frame_result_json, MovingAverage& moving_average, std::shared_ptr<AbstractSkeletonFilterBuilder<double>> filter_builder);
void processThread(k4a_calibration_t sensor_calibration, std::promise<std::tuple<nlohmann::json, PlotWrap<double>>>, std::shared_ptr<AbstractSkeletonFilterBuilder<double>> filter_builder, Benchmark& bench);

std::optional<Samples::Plane> detect_floor(MeasuredFrame frame, k4a_calibration_t sensor_calibration, Samples::PointCloudGenerator& pointCloudGenerator, Samples::FloorDetector& floorDetector, nlohmann::json& frame_result_json, MovingAverage& moving_average);
void apply_filter(
    MeasuredFrame frame,
    k4a_calibration_t sensor_calibration,
    Samples::FloorDetector& floorDetector,
    std::map<uint32_t, CurrentFilterType>& filters,
    nlohmann::json& frame_result_json,
    MovingAverage& moving_average,
    std::shared_ptr<AbstractSkeletonFilterBuilder<double>> builder,
    Benchmark& bench);
