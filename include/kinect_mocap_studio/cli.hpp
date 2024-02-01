#pragma once
#include <string>

#include <k4a/k4atypes.h>
#include <k4arecord/record.h>
#include <filter/AbstractSkeletonFilter.hpp>

struct CliConfig {
    public:
    std::string output_file_name;
    double temporal_smoothing = 0.;
    double measurement_error_factor = 5.0;
    // bool save_camera_data = false;
    int k4a_depth_mode = 0;
    std::string k4a_depth_mode_str;
    std::shared_ptr<AbstractSkeletonFilterBuilder<double>> filter_builder;
    std::string kalman_filter_type_str;
    std::string input_sensor_file_str;
    bool process_sensor_file = false;
    int k4a_frames_per_second = 0;
    std::string k4a_color_resolution_str;
    int k4a_color_resolution = 0;
    bool record_sensor_data = false;

    std::string output_json_file;
    std::string output_sensor_file;

    CliConfig(int argc, char** argv);
    static void printAppUsage();
    void printConfig();

    void openDeviceOrRecording(k4a_device_t& device, k4a_playback_t& playback_handle, k4a_calibration_t& sensor_calibration, k4a_device_configuration_t& device_config);
};
