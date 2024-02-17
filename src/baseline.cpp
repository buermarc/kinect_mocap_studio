#include <iostream>
#include <k4a/k4a.h>
#include <k4a/k4atypes.h>
#include <kinect_mocap_studio/utils.hpp>
#include <fstream>
#include <nlohmann/json.hpp>
#include <tclap/CmdLine.h>

typedef std::chrono::high_resolution_clock hc;

int main(int argc, char** argv) {
    TCLAP::CmdLine cmd("baseline benchmark", ' ', "0.0");

    TCLAP::ValueArg<double> duration_arg("d", "duration",
        "How long the baseline should run", false, 5.0,
        "double");

    cmd.add(duration_arg);
    cmd.parse(argc, argv);

    double duration = duration_arg.getValue();
    nlohmann::json baseline_bench_json;
    std::vector<double> durations;
    bool running;

    k4a_device_t device;
    k4a_device_configuration_t device_config = K4A_DEVICE_CONFIG_INIT_DISABLE_ALL;
    device_config.color_resolution = K4A_COLOR_RESOLUTION_1080P;
    device_config.depth_mode = k4a_depth_mode_t(2);
    VERIFY(k4a_device_open(0, &device), "Open K4A Device failed!");
    VERIFY(k4a_device_start_cameras(device, &device_config), "Start K4A cameras failed!");
    k4a_capture_t sensor_capture = nullptr;

    auto total_start = hc::now();
    auto start = hc::now();


    while ((std::chrono::duration<double>(hc::now() - total_start)).count() < duration) {
        k4a_wait_result_t get_capture_result = k4a_device_get_capture(device, &sensor_capture, 60);
        if (get_capture_result == K4A_WAIT_RESULT_TIMEOUT) {
            std::cout << "error: k4a_device_get_capture() timed out" << std::endl;
        } else {
            std::cout << "error: k4a_device_get_capture(): " << get_capture_result << std::endl;
        }

        if (get_capture_result == K4A_WAIT_RESULT_SUCCEEDED) {
            std::cout << "Successfuly extracted" << std::endl;
            double millis = (std::chrono::duration<double, std::milli>(hc::now() - start)).count();
            durations.push_back(millis);

            // k4a_image_t color_image = k4a_capture_get_color_image(sensor_capture);
            // k4a_image_t depth_image = k4a_capture_get_depth_image(sensor_capture);


            // k4a_image_release(color_image);
            // k4a_image_release(depth_image);

            k4a_capture_release(sensor_capture);

            start = hc::now();
        }
    }
    baseline_bench_json["durations"] = durations;
    std::stringstream file_name;
    file_name << "bench/baseline.json";

    std::ofstream output_file(file_name.str());
    output_file << std::setw(4) << baseline_bench_json << std::endl;
    std::cout << "Baseline bench json written." << std::endl;

    k4a_device_stop_cameras(device);
    k4a_device_close(device);
}
