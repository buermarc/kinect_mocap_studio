#include <filesystem>
#include <fstream>
#include <kinect_mocap_studio/benchmark.hpp>
#include <nlohmann/json.hpp>

namespace fs = std::filesystem;

void Benchmark::save(std::string experiment)
{
    nlohmann::json json;
    json["camera"] = std::accumulate(camera.cbegin(), camera.cend(), 0.0) / camera.size();
    json["recording_body"] = std::accumulate(recording_body.cbegin(), recording_body.cend(), 0.0) / recording_body.size();
    json["recording_imu"] = std::accumulate(recording_imu.cbegin(), recording_imu.cend(), 0.0) / recording_imu.size();
    json["network"] = std::accumulate(network.cbegin(), network.cend(), 0.0) / network.size();
    json["imu"] = std::accumulate(imu.cbegin(), imu.cend(), 0.0) / imu.size();
    json["save_body"] = std::accumulate(save_body.cbegin(), save_body.cend(), 0.0) / save_body.size();
    json["save_imu"] = std::accumulate(save_imu.cbegin(), save_imu.cend(), 0.0) / save_imu.size();
    json["pointcloud"] = std::accumulate(pointcloud.cbegin(), pointcloud.cend(), 0.0) / pointcloud.size();
    json["measurement_queue_produce"] = std::accumulate(measurement_queue_produce.cbegin(), measurement_queue_produce.cend(), 0.0) / measurement_queue_produce.size();
    json["detect_floor"] = std::accumulate(detect_floor.cbegin(), detect_floor.cend(), 0.0) / detect_floor.size();
    json["apply_filter"] = std::accumulate(apply_filter.cbegin(), apply_filter.cend(), 0.0) / apply_filter.size();
    json["build_filter"] = std::accumulate(build_filter.cbegin(), build_filter.cend(), 0.0) / build_filter.size();
    json["step"] = std::accumulate(step.cbegin(), step.cend(), 0.0) / step.size();
    json["extract_stability_metrics"] = std::accumulate(extract_stability_metrics.cbegin(), extract_stability_metrics.cend(), 0.0) / extract_stability_metrics.size();
    json["process_queue_produce"] = std::accumulate(process_queue_produce.cbegin(), process_queue_produce.cend(), 0.0) / process_queue_produce.size();
    json["visualize"] = std::accumulate(visualize.cbegin(), visualize.cend(), 0.0) / visualize.size();

    if (!fs::is_directory("bench") || !fs::exists("bench")) { // Check if src folder existsb
        fs::create_directory("bench"); // create src folder
    }

    std::stringstream file_name;
    file_name << "bench/bench_" << experiment << ".json";

    std::ofstream output_file(file_name.str());
    output_file << std::setw(4) << json << std::endl;
}
