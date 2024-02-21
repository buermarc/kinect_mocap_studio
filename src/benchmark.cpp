#include <algorithm>
#include <filesystem>
#include <fstream>
#include <kinect_mocap_studio/benchmark.hpp>
#include <nlohmann/json.hpp>

namespace fs = std::filesystem;

double var(const std::vector<double>& vec, double mean) {
    auto size = vec.size();
    auto variance_func = [&mean, &size](double accumulator, const double& val) {
        return accumulator + ((val - mean)*(val - mean) / (size - 1));
    };
    return std::accumulate(vec.begin(), vec.end(), 0.0, variance_func);
}

void Benchmark::save(std::string experiment)
{
    nlohmann::json json;

    json["mean"]["camera"] = std::accumulate(camera.cbegin(), camera.cend(), 0.0) / camera.size();
    json["mean"]["recording_body"] = std::accumulate(recording_body.cbegin(), recording_body.cend(), 0.0) / recording_body.size();
    json["mean"]["recording_imu"] = std::accumulate(recording_imu.cbegin(), recording_imu.cend(), 0.0) / recording_imu.size();
    json["mean"]["network"] = std::accumulate(network.cbegin(), network.cend(), 0.0) / network.size();
    json["mean"]["imu"] = std::accumulate(imu.cbegin(), imu.cend(), 0.0) / imu.size();
    json["mean"]["save_body"] = std::accumulate(save_body.cbegin(), save_body.cend(), 0.0) / save_body.size();
    json["mean"]["save_imu"] = std::accumulate(save_imu.cbegin(), save_imu.cend(), 0.0) / save_imu.size();
    json["mean"]["pointcloud"] = std::accumulate(pointcloud.cbegin(), pointcloud.cend(), 0.0) / pointcloud.size();
    json["mean"]["measurement_queue_produce"] = std::accumulate(measurement_queue_produce.cbegin(), measurement_queue_produce.cend(), 0.0) / measurement_queue_produce.size();
    json["mean"]["detect_floor"] = std::accumulate(detect_floor.cbegin(), detect_floor.cend(), 0.0) / detect_floor.size();
    json["mean"]["apply_filter"] = std::accumulate(apply_filter.cbegin(), apply_filter.cend(), 0.0) / apply_filter.size();
    json["mean"]["build_filter"] = std::accumulate(build_filter.cbegin(), build_filter.cend(), 0.0) / build_filter.size();
    json["mean"]["step"] = std::accumulate(step.cbegin(), step.cend(), 0.0) / step.size();
    json["mean"]["extract_stability_metrics"] = std::accumulate(extract_stability_metrics.cbegin(), extract_stability_metrics.cend(), 0.0) / extract_stability_metrics.size();
    json["mean"]["process_queue_produce"] = std::accumulate(process_queue_produce.cbegin(), process_queue_produce.cend(), 0.0) / process_queue_produce.size();
    json["mean"]["visualize"] = std::accumulate(visualize.cbegin(), visualize.cend(), 0.0) / visualize.size();

    if (camera.size() != 0) {
        json["min"]["camera"] = *std::min_element(camera.cbegin(), camera.cend());
    } else {
        json["min"]["camera"] = nullptr;
    }
    if (recording_body.size() != 0) {
        json["min"]["recording_body"] = *std::min_element(recording_body.cbegin(), recording_body.cend());
    } else {
        json["min"]["recording_body"] = nullptr;
    }
    if (recording_imu.size() != 0) {
        json["min"]["recording_imu"] = *std::min_element(recording_imu.cbegin(), recording_imu.cend());
    } else {
        json["min"]["recording_imu"] = nullptr;
    }
    json["min"]["network"] = *std::min_element(network.cbegin(), network.cend());
    json["min"]["imu"] = *std::min_element(imu.cbegin(), imu.cend());
    json["min"]["save_body"] = *std::min_element(save_body.cbegin(), save_body.cend());
    json["min"]["save_imu"] = *std::min_element(save_imu.cbegin(), save_imu.cend());
    json["min"]["pointcloud"] = *std::min_element(pointcloud.cbegin(), pointcloud.cend());
    json["min"]["measurement_queue_produce"] = *std::min_element(measurement_queue_produce.cbegin(), measurement_queue_produce.cend());
    json["min"]["detect_floor"] = *std::min_element(detect_floor.cbegin(), detect_floor.cend());
    json["min"]["apply_filter"] = *std::min_element(apply_filter.cbegin(), apply_filter.cend());
    json["min"]["build_filter"] = *std::min_element(build_filter.cbegin(), build_filter.cend());
    json["min"]["step"] = *std::min_element(step.cbegin(), step.cend());
    json["min"]["extract_stability_metrics"] = *std::min_element(extract_stability_metrics.cbegin(), extract_stability_metrics.cend());
    json["min"]["process_queue_produce"] = *std::min_element(process_queue_produce.cbegin(), process_queue_produce.cend());
    json["min"]["visualize"] = *std::min_element(visualize.cbegin(), visualize.cend());

    if (camera.size() != 0) {
        json["max"]["camera"] = *std::max_element(camera.cbegin(), camera.cend());
    } else {
        json["max"]["camera"] = nullptr;
    }
    if (recording_body.size() != 0) {
        json["max"]["recording_body"] = *std::max_element(recording_body.cbegin(), recording_body.cend());
    } else {
        json["max"]["recording_body"] = nullptr;
    }
    if (recording_imu.size() != 0) {
        json["max"]["recording_imu"] = *std::max_element(recording_imu.cbegin(), recording_imu.cend());
    } else {
        json["max"]["recording_imu"] = nullptr;
    }
    json["max"]["network"] = *std::max_element(network.cbegin(), network.cend());
    json["max"]["imu"] = *std::max_element(imu.cbegin(), imu.cend());
    json["max"]["save_body"] = *std::max_element(save_body.cbegin(), save_body.cend());
    json["max"]["save_imu"] = *std::max_element(save_imu.cbegin(), save_imu.cend());
    json["max"]["pointcloud"] = *std::max_element(pointcloud.cbegin(), pointcloud.cend());
    json["max"]["measurement_queue_produce"] = *std::max_element(measurement_queue_produce.cbegin(), measurement_queue_produce.cend());
    json["max"]["detect_floor"] = *std::max_element(detect_floor.cbegin(), detect_floor.cend());
    json["max"]["apply_filter"] = *std::max_element(apply_filter.cbegin(), apply_filter.cend());
    json["max"]["build_filter"] = *std::max_element(build_filter.cbegin(), build_filter.cend());
    json["max"]["step"] = *std::max_element(step.cbegin(), step.cend());
    json["max"]["extract_stability_metrics"] = *std::max_element(extract_stability_metrics.cbegin(), extract_stability_metrics.cend());
    json["max"]["process_queue_produce"] = *std::max_element(process_queue_produce.cbegin(), process_queue_produce.cend());
    json["max"]["visualize"] = *std::max_element(visualize.cbegin(), visualize.cend());

    json["var"]["camera"] = var(camera, json["mean"]["camera"]);
    json["var"]["recording_body"] = var(recording_body, json["mean"]["recording_body"]);
    json["var"]["recording_imu"] = var(recording_imu, json["mean"]["recording_imu"]);
    json["var"]["network"] = var(network, json["mean"]["network"]);
    json["var"]["imu"] = var(imu, json["mean"]["imu"]);
    json["var"]["save_body"] = var(save_body, json["mean"]["save_body"]);
    json["var"]["save_imu"] = var(save_imu, json["mean"]["save_imu"]);
    json["var"]["pointcloud"] = var(pointcloud, json["mean"]["pointcloud"]);
    json["var"]["measurement_queue_produce"] = var(measurement_queue_produce, json["mean"]["measurement_queue_produce"]);
    json["var"]["detect_floor"] = var(detect_floor, json["mean"]["detect_floor"]);
    json["var"]["apply_filter"] = var(apply_filter, json["mean"]["apply_filter"]);
    json["var"]["build_filter"] = var(build_filter, json["mean"]["build_filter"]);
    json["var"]["step"] = var(step, json["mean"]["step"]);
    json["var"]["extract_stability_metrics"] = var(extract_stability_metrics, json["mean"]["extract_stability_metrics"]);
    json["var"]["process_queue_produce"] = var(process_queue_produce, json["mean"]["process_queue_produce"]);
    json["var"]["visualize"] = var(visualize, json["mean"]["visualize"]);

    if (!fs::is_directory("bench") || !fs::exists("bench")) { // Check if src folder existsb
        fs::create_directory("bench"); // create src folder
    }

    std::stringstream file_name;
    file_name << "bench/bench_" << experiment << ".json";

    std::ofstream output_file(file_name.str());
    output_file << std::setw(4) << json << std::endl;
}
