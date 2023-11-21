#pragma once
#include <boost/lockfree/spsc_queue.hpp>
#include <boost/atomic.hpp>

#include <kinect_mocap_studio/SafeQueue.hpp>

#include <optional>

#include <k4a/k4atypes.h>
#include <k4abttypes.h>

#include "FloorDetector.h"

#include <filter/Point.hpp>


extern boost::atomic<bool> s_isRunning;
extern boost::atomic<bool> s_visualizeJointFrame;
extern boost::atomic<int> s_layoutMode;


struct MeasuredFrame {
    // k4abt_frame_t body_frame;
    // k4a_image_t depth_image;
    k4a_imu_sample_t imu_sample;
    std::vector<Visualization::PointCloudVertex> cloudPoints;
    std::vector<uint16_t> depthBuffer;
    std::vector<Point<double>> joints;
    // std::optional<Samples::Plane> floor; // Can be calculated during processing via imu_sample which should be a safe struct

};

struct ProcessedFrame {
    std::vector<Visualization::PointCloudVertex> cloudPoints;
    std::vector<uint16_t> depthBuffer;
    std::vector<Point<double>> joints;
    std::optional<Samples::Plane> floor;
};

// Typedef boost::lockfree::spsc_queue<MeasuredFrame> MeasurementQueue;
// Typedef boost::lockfree::spsc_queue<ProcessedFrame> ProcessedQueue;

typedef SafeQueue<MeasuredFrame> MeasurementQueue;
typedef SafeQueue<ProcessedFrame> ProcessedQueue;

extern MeasurementQueue measurement_queue;
extern ProcessedQueue processed_queue;
