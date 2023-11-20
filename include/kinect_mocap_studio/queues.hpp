#pragma once
#include <boost/lockfree/spsc_queue.hpp>
#include <boost/atomic.hpp>

#include <kinect_mocap_studio/SafeQueue.hpp>

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

// Typedef boost::lockfree::spsc_queue<MeasuredFrame> MeasurementQueue;
// Typedef boost::lockfree::spsc_queue<ProcessedFrame> ProcessedQueue;

typedef SafeQueue<MeasuredFrame> MeasurementQueue;
typedef SafeQueue<ProcessedFrame> ProcessedQueue;

extern MeasurementQueue measurement_queue;
extern ProcessedQueue processed_queue;
