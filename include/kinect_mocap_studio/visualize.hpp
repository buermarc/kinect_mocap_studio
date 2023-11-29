#pragma once
#include <kinect_mocap_studio/queues.hpp>

#include <future>

#include <Window3dWrapper.h>
#include <k4a/k4atypes.h>
#include <k4abttypes.h>

#include <filter/SkeletonFilter.hpp>

void visualizeResult(k4abt_frame_t bodyFrame, Window3dWrapper& window3d, int depthWidth, int depthHeight, SkeletonFilterBuilder<double> builder, uint64_t timestamp);
void visualizeLogic(ProcessedFrame frame, std::vector<SkeletonFilter<double>>& filters);
void visualizeThread(k4a_calibration_t sensor_calibration, std::promise<nlohmann::json> filter_json_promise);
