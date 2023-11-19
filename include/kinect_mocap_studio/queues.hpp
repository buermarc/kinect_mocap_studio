#pragma once
#include <boost/lockfree/queue.hpp>
#include <boost/atomic.hpp>

#include <optional>

#include <k4a/k4atypes.h>
#include <k4abttypes.h>

#include "FloorDetector.h"


extern boost::atomic<bool> s_isRunning;
extern boost::atomic<bool> s_visualizeJointFrame;
extern boost::atomic<int> s_layoutMode;


struct MeasuredFrame {
    k4abt_frame_t body_frame;
    k4a_imu_sample_t imu_sample;
    k4a_image_t depth_image;

};

struct ProcessedFrame {
    double skeleton_data[32][3];
    k4a_image_t depth_image;
    std::optional<Samples::Plane> floor;

};

typedef boost::lockfree::queue<MeasuredFrame> MeasurementQueue;
extern MeasurementQueue measurement_queue;

typedef boost::lockfree::queue<ProcessedFrame> ProcessedQueue;
extern ProcessedQueue processed_queue;
