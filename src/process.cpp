#include <future>
#include <iostream>
#include <kinect_mocap_studio/moving_average.hpp>
#include <kinect_mocap_studio/plotwrap.hpp>
#include <kinect_mocap_studio/process.hpp>
#include <optional>
#include <thread>

#include <k4abt.h>

#include "FloorDetector.h"
#include "PointCloudGenerator.h"

#include <filter/ConstrainedSkeletonFilter.hpp>
#include <filter/Point.hpp>
#include <filter/SkeletonFilter.hpp>
#include <filter/adaptive/AdaptiveConstrainedSkeletonFilter.hpp>
#include <filter/adaptive/AdaptivePointFilter3D.hpp>
#include <filter/adaptive/AdaptiveZarchanFilter1D.hpp>
#include <filter/com.hpp>
#include <nlohmann/json.hpp>

#include <matplotlibcpp/matplotlibcpp.h>

namespace plt = matplotlibcpp;

#include <cassert>

#ifndef BENCH_PROCESS
#define BENCH_PROCESS 1
#endif

typedef std::chrono::high_resolution_clock hc;
typedef AdaptivePointFilter3D<double, AdaptiveZarchanFilter1D<double>> ZarPointFilter;
// typedef SkeletonFilter<double> CurrentFilterType;
typedef ConstrainedSkeletonFilter<double> CurrentFilterType;
// typedef AdaptiveConstrainedSkeletonFilter<double, ZarPointFilter> CurrentFilterType;
/*
 * For the FloorDetector:
 * Fit a plane to the depth points that are furthest away from
 * the camera in the direction of gravity (this will fail when the
 * camera accelerates by 0.2 m/s2 in any direction)
 * This uses code from teh floor_detector example code
 */

ConstrainedSkeletonFilterBuilder<double> builder(32);
// AdaptiveConstrainedSkeletonFilterBuilder<double, ZarPointFilter> builder(32, 2.0);

std::optional<Samples::Plane> detect_floor(MeasuredFrame frame, k4a_calibration_t sensor_calibration, Samples::FloorDetector& floorDetector, nlohmann::json& frame_result_json, MovingAverage& moving_average)
{
    // Get down-sampled cloud points.
    const int downsampleStep = 2;
    // Detect floor plane based on latest visual and inertial observations.
    const size_t minimumFloorPointCount = 1024 / (downsampleStep * downsampleStep);

    auto maybeFloorPlane = floorDetector.TryDetectFloorPlane(frame.cloudPoints, frame.imu_sample,
        sensor_calibration, minimumFloorPointCount);

    maybeFloorPlane = moving_average.get_average(maybeFloorPlane);

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

std::tuple<
    std::map<uint32_t, std::tuple<Point<double>, Point<double>, Plane<double>>>,
    std::map<uint32_t, std::vector<Point<double>>>,
    std::map<uint32_t, std::vector<Point<double>>>,
    std::vector<double>,
    std::map<uint32_t, Point<double>>>
apply_filter(
    MeasuredFrame& frame,
    std::map<uint32_t, CurrentFilterType>& filters)
{
    std::map<uint32_t, std::tuple<Point<double>, Point<double>, Plane<double>>> stability_properties;
    std::vector<double> durations;
    std::map<uint32_t, Point<double>> com_dots;
    std::map<uint32_t, std::vector<Point<double>>> fpositions;
    std::map<uint32_t, std::vector<Point<double>>> fvelocities;
    for (const auto& element : frame.joints) {
        uint32_t i = element.first;
        if (filters.find(i) == filters.end()) {
            filters.insert(std::map<uint32_t, CurrentFilterType>::value_type(i, builder.build()));
        }

        auto& filter = filters.at(i);
        if (!filter.is_initialized()) {
            filter.init(frame.joints.at(i), frame.timestamp);
            continue;
        }

        std::cout << "Duration: " << filter.time_diff(frame.timestamp) << std::endl;
        durations.push_back(filter.time_diff(frame.timestamp));
        auto [filtered_positions, filtered_velocities] = filter.step(frame.joints.at(i), frame.timestamp);

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
        stability_properties.insert(std::map<uint32_t, std::tuple<Point<double>, Point<double>, Plane<double>>>::value_type(i, std::make_tuple(com, xcom, bos_plane)));
        com_dots[i] = filter.calculate_com_dot();

        fpositions[i] = filtered_positions;
        fvelocities[i] = filtered_velocities;
    }
    return std::make_tuple(stability_properties, fpositions, fvelocities, durations, com_dots);
}

std::tuple<ProcessedFrame, PlottingFrame> processLogic(
    MeasuredFrame frame,
    k4a_calibration_t sensor_calibration,
    Samples::FloorDetector& floorDetector,
    std::map<uint32_t, CurrentFilterType>& filters,
    nlohmann::json& frame_result_json,
    MovingAverage& moving_average)
{
    // Can we detect the floor
    auto optional_point = detect_floor(frame, sensor_calibration, floorDetector, frame_result_json, moving_average);
    // Mutates joints
    auto [stability_properties, fpositions, fvelocities, durations, com_dots] = apply_filter(frame, filters);
    auto filtered_joints(fpositions);

    return std::make_tuple(
        ProcessedFrame { frame.imu_sample, std::move(frame.cloudPoints), std::move(fpositions), std::move(frame.confidence_levels), std::move(stability_properties), optional_point, com_dots },
        PlottingFrame { std::move(frame.joints), std::move(filtered_joints), std::move(fvelocities), std::move(durations), com_dots });
}

void processThread(
    k4a_calibration_t sensor_calibration,
    std::promise<std::tuple<nlohmann::json, PlotWrap<double>>> process_json_promise)
{
    Samples::PointCloudGenerator pointCloudGenerator { sensor_calibration };
    Samples::FloorDetector floorDetector;
    MeasuredFrame frame;
    std::map<uint32_t, CurrentFilterType> filters;
    MovingAverage moving_average(40);

    nlohmann::json frame_result_json;

    while (s_isRunning) {
#ifdef BENCH_PROCESS
        auto start = hc::now();
        auto latency = hc::now();
#endif
        bool retrieved = measurement_queue.Consume(frame);
        if (retrieved) {
            auto start = hc::now();
            auto [processed_frame, plotting_frame] = processLogic(frame, sensor_calibration, floorDetector, filters, frame_result_json, moving_average);
            processed_queue.Produce(std::move(processed_frame));
            plotting_queue.Produce(std::move(plotting_frame));

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

    PlotWrap<double> plotwrap;
    /*
    if (filters.size() > 0) {
        auto filter = filters.cbegin()->second;
        auto unpositions = filter.get_unfiltered_positions();
        auto positions = filter.get_filtered_positions();
        auto velocities = filter.get_filtered_velocities();
        auto timestamps = filter.get_timestamps();
        assert(timestamps.size() == positions.size());
        std::cout << "Timestamp size: " << timestamps.size() << std::endl;
        std::cout << "Positions size: " << positions.size() << std::endl;

        std::vector<double> unx(positions.size());
        std::vector<double> uny(positions.size());
        std::vector<double> unz(positions.size());

        std::vector<double> x(positions.size());
        std::vector<double> y(positions.size());
        std::vector<double> z(positions.size());

        std::transform(positions.cbegin(), positions.cend(), x.begin(), [](auto ele) { return ele.at(HAND_RIGHT).x; });
        std::transform(positions.cbegin(), positions.cend(), y.begin(), [](auto ele) { return ele.at(HAND_RIGHT).y; });
        std::transform(positions.cbegin(), positions.cend(), z.begin(), [](auto ele) { return ele.at(HAND_RIGHT).z; });

        std::transform(unpositions.cbegin(), unpositions.cend(), unx.begin(), [](auto ele) { return ele.at(HAND_RIGHT).x; });
        std::transform(unpositions.cbegin(), unpositions.cend(), uny.begin(), [](auto ele) { return ele.at(HAND_RIGHT).y; });
        std::transform(unpositions.cbegin(), unpositions.cend(), unz.begin(), [](auto ele) { return ele.at(HAND_RIGHT).z; });

        std::vector<double> vel_x(velocities.size());
        std::vector<double> vel_y(velocities.size());
        std::vector<double> vel_z(velocities.size());

        std::transform(velocities.cbegin(), velocities.cend(), vel_x.begin(), [](auto ele) { return ele.at(HAND_RIGHT).x; });
        std::transform(velocities.cbegin(), velocities.cend(), vel_y.begin(), [](auto ele) { return ele.at(HAND_RIGHT).y; });
        std::transform(velocities.cbegin(), velocities.cend(), vel_z.begin(), [](auto ele) { return ele.at(HAND_RIGHT).z; });


        // std::vector<double> diff_x;
        // diff_x.reserve(positions.size());
        // diff_x.push_back(0);
        // for (int i = 0; i < positions.size() - 1; ++i) {
        //     auto position_n = positions.at(i).at(HAND_RIGHT);
        //     auto position_n1 = positions.at(i+1).at(HAND_RIGHT);
        //     auto timestamp_n = timestamps.at(i);
        //     auto timestamp_n1 = timestamps.at(i+1);
        //     diff_x.push_back((position_n1.x - position_n.x) / (timestamp_n1 - timestamp_n));
        // }

        // std::vector<double> diff_y;
        // diff_y.reserve(positions.size());
        // diff_y.push_back(0);
        // for (int i = 0; i < positions.size() - 1; ++i) {
        //     auto position_n = positions.at(i).at(HAND_RIGHT);
        //     auto position_n1 = positions.at(i+1).at(HAND_RIGHT);
        //     auto timestamp_n = timestamps.at(i);
        //     auto timestamp_n1 = timestamps.at(i+1);
        //     diff_y.push_back((position_n1.y - position_n.y) / (timestamp_n1 - timestamp_n));
        // }

        // std::vector<double> diff_z;
        // diff_z.reserve(positions.size());
        // diff_z.push_back(0);
        // for (int i = 0; i < positions.size() - 1; ++i) {
        //     auto position_n = positions.at(i).at(HAND_RIGHT);
        //     auto position_n1 = positions.at(i+1).at(HAND_RIGHT);
        //     auto timestamp_n = timestamps.at(i);
        //     auto timestamp_n1 = timestamps.at(i+1);
        //     diff_z.push_back((position_n1.z - position_n.z) / (timestamp_n1 - timestamp_n));
        // }

        // plt::title("Hand Right Velocities X");
        // plt::named_plot("Finite diff x", timestamps, diff_x);
        // plt::named_plot("Filter velocity x", timestamps, vel_x);
        // plt::legend();
        // plt::show(true);
        // plt::cla();

        // plt::title("Hand Right Velocities Y");
        // plt::named_plot("Finite diff y", timestamps, diff_y);
        // plt::named_plot("Filter velocity y", timestamps, vel_x);
        // plt::legend();
        // plt::show(true);
        // plt::cla();

        // plt::title("Hand Right Velocities Z ");
        // plt::named_plot("Finite diff z", timestamps, diff_z);
        // plt::named_plot("Filter velocity z", timestamps, vel_x);
        // plt::legend();
        // plt::show(true);
        // plt::cla();
        //


        NamedPlot<double> plot_a("Hand Right Position X", true);
        plot_a.add_line(NamedLine<double>("Unfiltered", timestamps, unx));
        plot_a.add_line(NamedLine<double>("Filtered", timestamps, x));
        plotwrap.add_named_plot(plot_a);

        NamedPlot<double> plot_b("Hand Right Position Y", true);
        plot_b.add_line(NamedLine<double>("Unfiltered", timestamps, uny));
        plot_b.add_line(NamedLine<double>("Filtered", timestamps, y));
        plotwrap.add_named_plot(plot_b);

        NamedPlot<double> plot_c("Hand Right Position Z", true);
        plot_c.add_line(NamedLine<double>("Unfiltered", timestamps, unz));
        plot_c.add_line(NamedLine<double>("Filtered", timestamps, z));
        plotwrap.add_named_plot(plot_c);
    }
    //plt::close();
    */
    std::cout << "Process Thread Exit" << std::endl;
    s_stillPlotting = false;

    frame_result_json["filters"] = filters;
    process_json_promise.set_value(std::make_tuple(frame_result_json, plotwrap));
}
