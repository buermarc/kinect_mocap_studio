#include <kinect_mocap_studio/process.hpp>
#include <optional>
#include <iostream>
#include <thread>
#include <future>

#include <k4abt.h>

#include "PointCloudGenerator.h"
#include "FloorDetector.h"

#include <nlohmann/json.hpp>
#include <filter/com.hpp>
#include <filter/Point.hpp>
#include <filter/SkeletonFilter.hpp>
#include <filter/adaptive/AdaptiveConstrainedSkeletonFilter.hpp>
#include <filter/adaptive/AdaptivePointFilter3D.hpp>
#include <filter/adaptive/AdaptiveZarchanFilter1D.hpp>

#include <matplotlibcpp/matplotlibcpp.h>

namespace plt = matplotlibcpp;

#include <cassert>


#ifndef BENCH_PROCESS
#define BENCH_PROCESS 1
#endif

typedef std::chrono::high_resolution_clock hc;
typedef AdaptivePointFilter3D<double, AdaptiveZarchanFilter1D<double>> ZarPointFilter;
typedef SkeletonFilter<double> CurrentFilterType;
// typedef AdaptiveConstrainedSkeletonFilter<double, ZarPointFilter> CurrentFilterType;
/*
 * For the FloorDetector:
 * Fit a plane to the depth points that are furthest away from
 * the camera in the direction of gravity (this will fail when the
 * camera accelerates by 0.2 m/s2 in any direction)
 * This uses code from teh floor_detector example code
 */

SkeletonFilterBuilder<double> builder(32, 2.0);
// AdaptiveConstrainedSkeletonFilterBuilder<double, ZarPointFilter> builder(32, 2.0);

std::optional<Samples::Plane> detect_floor(MeasuredFrame frame, k4a_calibration_t sensor_calibration, Samples::FloorDetector& floorDetector, nlohmann::json& frame_result_json) {
    // Get down-sampled cloud points.
    const int downsampleStep = 2;
    // Detect floor plane based on latest visual and inertial observations.
    const size_t minimumFloorPointCount = 1024 / (downsampleStep * downsampleStep);

    const auto& maybeFloorPlane = floorDetector.TryDetectFloorPlane(frame.cloudPoints, frame.imu_sample,
        sensor_calibration, minimumFloorPointCount);


    nlohmann::json floor_result_json;
    if (maybeFloorPlane.has_value()) {
        // For visualization purposes, make floor origin the projection of a point 1.5m in front of the camera.
        Samples::Vector cameraOrigin = { 0, 0, 0 };
        Samples::Vector cameraForward = { 0, 0, 1 };

        auto p = maybeFloorPlane->ProjectPoint(cameraOrigin)
            + maybeFloorPlane->ProjectVector(cameraForward) * 1.5f;

        auto n = maybeFloorPlane->Normal;

        floor_result_json["point"].push_back(
            { p.X * 1000.f, p.Y * 1000.f, p.Z * 1000.f });
        floor_result_json["normal"].push_back({ n.X, n.Y, n.Z });
        floor_result_json["valid"] = true;
    } else {
        floor_result_json["point"].push_back({ 0., 0., 0. });
        floor_result_json["normal"].push_back({ 0., 0., 0. });
        floor_result_json["valid"] = false;
    }
    frame_result_json["floor"].push_back(floor_result_json);

    return maybeFloorPlane;
}

std::vector<std::tuple<Point<double>, Point<double>, Plane<double>>>
apply_filter(
    MeasuredFrame& frame,
    std::vector<CurrentFilterType>& filters
) {
    std::vector<std::tuple<Point<double>, Point<double>, Plane<double>>> stability_properties;
    for (int i = 0; i < frame.joints.size(); ++i)
    {
        if (filters.empty() or filters.size() <= frame.joints.size()) {
            filters.push_back(builder.build());
        }

        auto& filter = filters.at(i);
        if (!filter.is_initialized()) {
            filter.init(frame.joints.at(i), frame.timestamp);
            continue;
        }

        auto [filtered_positions, _ ] = filter.step(frame.joints.at(i), frame.timestamp);

        auto com = filter.calculate_com();
        auto ankle_left = filtered_positions[ANKLE_LEFT];
        auto ankle_right = filtered_positions[ANKLE_RIGHT];

        // Take point in the middle of both ankles
        Point<double> mean_ankle;
        mean_ankle.x = (ankle_left.x + ankle_right.x) / 2;
        mean_ankle.y = (ankle_left.y + ankle_right.y) / 2;
        mean_ankle.z = (ankle_left.z + ankle_right.z) / 2;

        // Calc euclidean norm from mean to com
        auto ankle_com_norm = std::sqrt(
            std::pow(mean_ankle.x - com.x, 2) + std::pow(mean_ankle.y - com.y, 2) + std::pow(mean_ankle.z - com.z, 2));

        auto xcom = filter.calculate_x_com(ankle_com_norm);
        Plane<double> bos_plane = azure_kinect_bos(filtered_positions);
        stability_properties.push_back(std::make_tuple(com, xcom, bos_plane));

        frame.joints.at(i) = std::move(filtered_positions);
    }
    return stability_properties;
}

ProcessedFrame processLogic(
    MeasuredFrame frame,
    k4a_calibration_t sensor_calibration,
    Samples::FloorDetector& floorDetector,
    std::vector<CurrentFilterType>& filters,
    nlohmann::json& frame_result_json
) {
    // Can we detect the floor
    auto optional_point = detect_floor(frame, sensor_calibration, floorDetector, frame_result_json);
    // Mutates joints
    auto stability_properties = apply_filter(frame, filters);

    return ProcessedFrame { frame.imu_sample, frame.cloudPoints, frame.joints, frame.confidence_levels, stability_properties, optional_point };
}

void processThread(
    k4a_calibration_t sensor_calibration,
    std::promise<nlohmann::json> process_json_promise
) {
    Samples::PointCloudGenerator pointCloudGenerator { sensor_calibration };
    Samples::FloorDetector floorDetector;
    MeasuredFrame frame;
    std::vector<CurrentFilterType> filters;

    nlohmann::json frame_result_json;

    while (s_isRunning) {
#ifdef BENCH_PROCESS
        auto start = hc::now();
        auto latency = hc::now();
#endif
        bool retrieved = measurement_queue.Consume(frame);
        if (retrieved) {
            auto start = hc::now();
            ProcessedFrame result = processLogic(frame, sensor_calibration, floorDetector, filters, frame_result_json);
            processed_queue.Produce(std::move(result));

#ifdef BENCH_PROCESS
            auto stop = hc::now();
            std::chrono::duration<double, std::milli> time = stop - start;
            std::chrono::duration<double, std::milli> latency_duration = stop - latency;
            std::cout << "Process Duration: " << time.count() << "ms\n";
            std::cout << "Latency Process: " << latency_duration.count() << "ms\n";
#endif
        } else {
            std::this_thread::yield();
        }
    }

    // Idea, plot finite diff against velocities of filter
    frame_result_json["filters"] = filters;
    process_json_promise.set_value(frame_result_json);
    if (filters.size() > 0) {
        auto filter = filters.at(0);
        auto positions = filter.get_unfiltered_positions();
        auto velocities = filter.get_filtered_velocities();
        auto timestamps = filter.get_timestamps();
        assert(timestamps.size() == positions.size());
        std::cout << "Timestamp size: " << timestamps.size() << std::endl;
        std::cout << "Positions size: " << positions.size() << std::endl;

        std::vector<double> x(positions.size());
        std::vector<double> y(positions.size());
        std::vector<double> z(positions.size());
        std::transform(positions.cbegin(), positions.cend(), x.begin(), [](auto ele) { return ele.at(HAND_RIGHT).x; });
        std::transform(positions.cbegin(), positions.cend(), y.begin(), [](auto ele) { return ele.at(HAND_RIGHT).y; });
        std::transform(positions.cbegin(), positions.cend(), z.begin(), [](auto ele) { return ele.at(HAND_RIGHT).z; });

        std::vector<double> vel_x(velocities.size());
        std::vector<double> vel_y(velocities.size());
        std::vector<double> vel_z(velocities.size());

        std::transform(velocities.cbegin(), velocities.cend(), vel_x.begin(), [](auto ele) { return ele.at(HAND_RIGHT).x; });
        std::transform(velocities.cbegin(), velocities.cend(), vel_y.begin(), [](auto ele) { return ele.at(HAND_RIGHT).y; });
        std::transform(velocities.cbegin(), velocities.cend(), vel_z.begin(), [](auto ele) { return ele.at(HAND_RIGHT).z; });


        std::vector<double> diff_x;
        diff_x.reserve(positions.size());
        diff_x.push_back(0);
        for (int i = 0; i < positions.size() - 1; ++i) {
            auto position_n = positions.at(i).at(HAND_RIGHT);
            auto position_n1 = positions.at(i+1).at(HAND_RIGHT);
            auto timestamp_n = timestamps.at(i);
            auto timestamp_n1 = timestamps.at(i+1);
            diff_x.push_back((position_n1.x - position_n.x) / (timestamp_n1 - timestamp_n));
        }

        std::cout << "diff_x size: " << diff_x.size() << std::endl;
        plt::named_plot("diff_x", timestamps, diff_x);
        plt::named_plot("vel_x", timestamps, vel_x);
        // plt::named_plot("y", timestamps, y);
        // plt::named_plot("z", timestamps, z);
        plt::title("Test");
        plt::legend();
        plt::show(true);
    }
    plt::close();
    std::cout << "Process Thread Exit" << std::endl;
}
