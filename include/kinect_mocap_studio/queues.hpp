#pragma once
#include <boost/atomic.hpp>

#include <kinect_mocap_studio/SafeQueue.hpp>

#include <vector>
#include <optional>

#include <k4a/k4atypes.h>
#include <k4abttypes.h>

#include "FloorDetector.h"

#include <filter/Point.hpp>
#include <filter/com.hpp>


extern boost::atomic<bool> s_stillPlotting;
extern boost::atomic<bool> s_isRunning;
extern boost::atomic<bool> s_visualizeJointFrame;
extern boost::atomic<int> s_layoutMode;
extern boost::atomic<bool> s_glfwInitialized;


struct MeasuredFrame {
    // k4abt_frame_t body_frame;
    // k4a_image_t depth_image;
    k4a_imu_sample_t imu_sample;
    std::vector<k4a_float3_t> cloudPoints;
    std::vector<std::vector<Point<double>>> joints;
    std::vector<std::vector<int>> confidence_levels;
    double timestamp;
    // std::optional<Samples::Plane> floor; // Can be calculated during processing via imu_sample which should be a safe struct

};

struct ProcessedFrame {
    k4a_imu_sample_t imu_sample;
    std::vector<k4a_float3_t> cloudPoints;
    std::vector<std::vector<Point<double>>> joints;
    std::vector<std::vector<int>> confidence_levels;
    std::vector<std::tuple<Point<double>, Point<double>, Plane<double>>> stability_properties;
    std::optional<Samples::Plane> floor;
};

struct PlottingFrame {
    std::vector<std::vector<Point<double>>> unfiltered_joints;
    std::vector<std::vector<Point<double>>> filtered_joints;
    std::vector<std::vector<Point<double>>> filtered_vel;
    std::vector<double> durations;
    std::vector<Point<double>> com_dots;
};

// Typedef boost::lockfree::spsc_queue<MeasuredFrame> MeasurementQueue;
// Typedef boost::lockfree::spsc_queue<ProcessedFrame> ProcessedQueue;

typedef SafeQueue<MeasuredFrame> MeasurementQueue;
typedef SafeQueue<ProcessedFrame> ProcessedQueue;
typedef SafeQueue<PlottingFrame> PlottingQueue;

extern MeasurementQueue measurement_queue;
extern ProcessedQueue processed_queue;
extern PlottingQueue plotting_queue;
