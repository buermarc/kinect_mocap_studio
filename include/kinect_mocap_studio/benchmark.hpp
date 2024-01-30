#pragma once
#include <vector>
#include <string>

class Benchmark {
    public:
    std::vector<double> camera;
    std::vector<double> recording_body;
    std::vector<double> recording_imu;
    std::vector<double> network;
    std::vector<double> imu;
    std::vector<double> save_body;
    std::vector<double> save_imu;
    std::vector<double> pointcloud;
    std::vector<double> measurement_queue_produce;

    std::vector<double> detect_floor;
    std::vector<double> apply_filter;
    std::vector<double> build_filter;
    std::vector<double> step;
    std::vector<double> extract_stability_metrics;
    std::vector<double> process_queue_produce;

    std::vector<double> visualize;

    void save(std::string experiment);
};
