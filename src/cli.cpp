#include <kinect_mocap_studio/cli.hpp>
#include <kinect_mocap_studio/utils.hpp>

#include <string>
#include <cstdint>
#include <fstream>

#include <k4a/k4a.h>
#include <k4abttypes.h>
#include <k4a/k4atypes.h>
#include <k4arecord/playback.h>
#include <k4arecord/record.h>
#include <tclap/CmdLine.h>

void CliConfig::printAppUsage()
{
    printf("\n");
    printf(" Basic Navigation:\n\n");
    printf(" Rotate: Rotate the camera by moving the mouse while holding mouse"
           " left button\n");
    printf(" Pan: Translate the scene by holding Ctrl key and drag the scene "
           "with mouse left button\n");
    printf(" Zoom in/out: Move closer/farther away from the scene center by "
           "scrolling the mouse scroll wheel\n");
    printf(" Select Center: Center the scene based on a detected joint by "
           "right clicking the joint with mouse\n");
    printf("\n");
    printf(" Key Shortcuts\n\n");
    printf(" ESC: quit\n");
    printf(" h: help\n");
    printf(" b: body visualization mode\n");
    printf(" k: 3d window layout\n");
    printf("\n");
}

CliConfig::CliConfig(int argc, char** argv)
{
    try {

        TCLAP::CmdLine cmd("kinect_mocap_studio is a command-line tool to record"
                           "video and skeletal data from the Azure-Kinect",
            ' ', "0.0");

        TCLAP::ValueArg<std::string> input_sensor_file_arg("i", "infile",
            "Input sensor file of type *.mkv to process", false, "",
            "string");

        cmd.add(input_sensor_file_arg);

        TCLAP::ValueArg<bool> record_sensor_data_arg("w", "write",
            "Write sensor data to *.mkv file", false, false,
            "bool");

        cmd.add(record_sensor_data_arg);

        TCLAP::ValueArg<std::string> output_name_arg("o", "outfile",
            "Name of output file excluding the file extension", false, "output",
            "string");

        cmd.add(output_name_arg);

        TCLAP::ValueArg<std::string> depth_mode_arg("d", "depth_mode",
            "Depth mode: OFF, NFOV_2X2BINNED, NFOV_UNBINNED, "
            "WFOV_2X2BINNED, WFOV_UNBINNED, PASSIVE_IR",
            false,
            "NFOV_UNBINNED", "string");

        cmd.add(depth_mode_arg);

        TCLAP::ValueArg<std::string> k4a_color_resolution_arg("c", "color_mode",
            "Color resolution: OFF, 720P, 1080P, "
            "1440P, 1536P, 2160P, 3072P",
            false,
            "OFF", "string");

        cmd.add(k4a_color_resolution_arg);

        TCLAP::ValueArg<int> k4a_frames_per_second_arg("f", "fps",
            "Frames per second: 5, 15, 30", false,
            30, "int");

        cmd.add(k4a_frames_per_second_arg);

        TCLAP::ValueArg<double> temporal_smoothing_arg("s", "smoothing",
            "Amount of temporal smoothing in the skeleton tracker (0-1)", false,
            K4ABT_DEFAULT_TRACKER_SMOOTHING_FACTOR, "double");

        cmd.add(temporal_smoothing_arg);

        // TCLAP::SwitchArg mkv_switch("m","mkv","Record rgb and depth camera data"
        //                               " to an *.mkv file", cmd, false);

        // Parse the argv array.
        cmd.parse(argc, argv);

        // Get the value parsed by each arg.
        output_file_name = output_name_arg.getValue();

        k4a_depth_mode_str = depth_mode_arg.getValue();
        if (std::strcmp(k4a_depth_mode_str.c_str(), "OFF") == 0) {
            k4a_depth_mode = 0;
        } else if (std::strcmp(k4a_depth_mode_str.c_str(), "NFOV_2X2BINNED") == 0) {
            k4a_depth_mode = 1;
        } else if (std::strcmp(k4a_depth_mode_str.c_str(), "NFOV_UNBINNED") == 0) {
            k4a_depth_mode = 2;
        } else if (std::strcmp(k4a_depth_mode_str.c_str(), "WFOV_2X2BINNED") == 0) {
            k4a_depth_mode = 3;
        } else if (std::strcmp(k4a_depth_mode_str.c_str(), "WFOV_UNBINNED") == 0) {
            k4a_depth_mode = 4;
        } else if (std::strcmp(k4a_depth_mode_str.c_str(), "PASSIVE_IR") == 0) {
            k4a_depth_mode = 5;
        } else {
            std::cerr << "error: depth_mode must be: OFF, NFOV_2X2BINNED, "
                      << "NFOV_UNBINNED, WFOV_2X2BINNED, WFOV_UNBINNED, PASSIVE_IR."
                      << std::endl;
            exit(1);
        }

        k4a_color_resolution_str = k4a_color_resolution_arg.getValue();
        if (std::strcmp(k4a_color_resolution_str.c_str(), "OFF") == 0) {
            k4a_color_resolution = 0;
        } else if (std::strcmp(k4a_color_resolution_str.c_str(), "720P") == 0) {
            k4a_color_resolution = 1;
        } else if (std::strcmp(k4a_color_resolution_str.c_str(), "1080P") == 0) {
            k4a_color_resolution = 2;
        } else if (std::strcmp(k4a_color_resolution_str.c_str(), "1440P") == 0) {
            k4a_color_resolution = 3;
        } else if (std::strcmp(k4a_color_resolution_str.c_str(), "1536P") == 0) {
            k4a_color_resolution = 4;
        } else if (std::strcmp(k4a_color_resolution_str.c_str(), "2160P") == 0) {
            k4a_color_resolution = 5;
        } else if (std::strcmp(k4a_color_resolution_str.c_str(), "3072P") == 0) {
            k4a_color_resolution = 6;
        } else {
            std::cerr << "error: color resolution must be: OFF, 720P, 1080P, "
                         "1440P, 1536P, 2160P, 3072P"
                      << std::endl;
            exit(1);
        }

        record_sensor_data = record_sensor_data_arg.getValue();

        k4a_frames_per_second = k4a_frames_per_second_arg.getValue();
        if (k4a_frames_per_second != 5
            && k4a_frames_per_second != 15
            && k4a_frames_per_second != 30) {
            std::cerr << "error: fps must be 5, 15, or 30"
                      << std::endl;
            exit(1);
        }

        temporal_smoothing = temporal_smoothing_arg.getValue();
        if (temporal_smoothing > 1.0 || temporal_smoothing < 0.0) {
            std::cerr << "error: temporal_smoothing must be between 0.0-1.0"
                      << std::endl;
            exit(1);
        }

        input_sensor_file_str = input_sensor_file_arg.getValue();
        if (input_sensor_file_str.length() > 0) {
            std::string mkv_ext = ".mkv";
            if (input_sensor_file_str.find(mkv_ext) != (input_sensor_file_str.length() - 4)) {
                std::cerr << "error: input sensor file must be of type *.mkv"
                          << std::endl;
                exit(1);
            }
            process_sensor_file = true;
            output_file_name = input_sensor_file_str.substr(0, input_sensor_file_str.length() - 4);
        }

        if (process_sensor_file && record_sensor_data) {
            std::cerr << "error: cannot process an input file (-i) and write"
                         "(-w) the sensor data at the same time"
                      << std::endl;
            exit(1);
        }

    }
    catch (TCLAP::ArgException& e)
    {
    std::cerr << "error: " << e.error() << " for arg " << e.argId()
              << std::endl;
    exit(1);
    }

    CliConfig::printAppUsage();

    /// From here we start the real application

    int frame_count = 0;
    // int frame_count_max   = 100;

    // It appears as though some of the configuration information is
    // not written to the mkv files and so can become lost. Here I'm
    // using an ugly but practical solution of embedding this meta data in
    // the file name.
    if (!process_sensor_file) {
        std::stringstream output_file_name_ss;
        output_file_name_ss << output_file_name;
        output_file_name_ss << "_" << k4a_depth_mode_str
                            << "_" << k4a_color_resolution_str
                            << "_" << k4a_frames_per_second << "fps";
        output_file_name = output_file_name_ss.str();
    }

    output_json_file = output_file_name + ".json";
    output_sensor_file = output_file_name + ".mkv";

    // Check to make sure the file name is unique
    int counter = 0;
    bool name_collision = false;
    do {
        std::ifstream jsonFile(output_json_file.c_str());
        std::ifstream sensorFile(output_sensor_file.c_str());
        name_collision = false;

        if (jsonFile.good()) {
            name_collision = true;
            jsonFile.close();
        }
        if (sensorFile.good()) {
            name_collision = true;
            sensorFile.close();
        }
        if (name_collision) {
            std::stringstream ss;
            ss << output_file_name << "_" << counter;
            output_json_file = ss.str() + ".json";
            output_sensor_file = ss.str() + ".mkv";
            counter++;
        }

    } while (name_collision);

}

void CliConfig::openDeviceOrRecording(k4a_device_t& device, k4a_playback_t& playback_handle, k4a_calibration_t& sensor_calibration, k4a_device_configuration_t& device_config)
{
    if (process_sensor_file) {
        VERIFY(k4a_playback_open(input_sensor_file_str.c_str(), &playback_handle),
            "error: k4a_playback_open() failed");
        VERIFY(k4a_playback_get_calibration(playback_handle, &sensor_calibration),
            "error: k4a_playback_get_calibration() failed");

        k4a_record_configuration_t record_config;
        VERIFY(k4a_playback_get_record_configuration(playback_handle, &record_config),
            "error: k4a_playback_get_record_configuration() failed");

        switch (record_config.depth_mode) {
        case K4A_DEPTH_MODE_OFF: {
            k4a_depth_mode_str = "OFF";
        } break;
        case K4A_DEPTH_MODE_NFOV_2X2BINNED: {
            k4a_depth_mode_str = "NFOV_2X2BINNED";
        } break;
        case K4A_DEPTH_MODE_NFOV_UNBINNED: {
            k4a_depth_mode_str = "NFOV_UNBINNED";
        } break;
        case K4A_DEPTH_MODE_WFOV_2X2BINNED: {
            k4a_depth_mode_str = "WFOV_2X2BINNED";
        } break;
        case K4A_DEPTH_MODE_WFOV_UNBINNED: {
            k4a_depth_mode_str = "WFOV_UNBINNED";
        } break;
        case K4A_DEPTH_MODE_PASSIVE_IR: {
            k4a_depth_mode_str = "PASSIVE_IR";
        } break;
        default: {
            std::cerr << "error: unrecognized depth_mode in recording" << std::endl;
        }
        };

        switch (record_config.color_resolution) {
        case K4A_COLOR_RESOLUTION_OFF: {
            k4a_color_resolution_str = "OFF";
        } break;
        case K4A_COLOR_RESOLUTION_720P: {
            k4a_color_resolution_str = "720P";
        } break;
        case K4A_COLOR_RESOLUTION_1080P: {
            k4a_color_resolution_str = "1080P";
        } break;
        case K4A_COLOR_RESOLUTION_1440P: {
            k4a_color_resolution_str = "1440P";
        } break;
        case K4A_COLOR_RESOLUTION_1536P: {
            k4a_color_resolution_str = "1536P";
        } break;
        case K4A_COLOR_RESOLUTION_2160P: {
            k4a_color_resolution_str = "2160P";
        } break;
        case K4A_COLOR_RESOLUTION_3072P: {
            k4a_color_resolution_str = "3072P";
        } break;
        default: {
            std::cerr << "error: unrecognized color resolution in recording" << std::endl;
        }
        };

        k4a_frames_per_second = record_config.camera_fps;

        // device_config.camera_fps = record_config.camera_fps;
        // device_config.color_format = record_config.color_format;
        // device_config.color_resolution = record_config.color_resolution;
        // device_config.depth_delay_off_color_usec = record_config.depth_delay_off_color_usec;
        // device_config.depth_mode = record_config.depth_mode;
        // device_config.subordinate_delay_off_master_usec = record_config.subordinate_delay_off_master_usec;
        // device_config.wired_sync_mode = record_config.wired_sync_mode;

    } else {

        VERIFY(k4a_device_open(0, &device), "Open K4A Device failed!");
        // Start camera. Make sure depth camera is enabled.
        device_config.depth_mode = k4a_depth_mode_t(k4a_depth_mode);

        switch (k4a_frames_per_second) {
        case 5: {
            device_config.camera_fps = K4A_FRAMES_PER_SECOND_5;
        } break;
        case 15: {
            device_config.camera_fps = K4A_FRAMES_PER_SECOND_15;
        } break;
        case 30: {
            device_config.camera_fps = K4A_FRAMES_PER_SECOND_30;
        } break;
        default: {
            std::cerr << "error: fps must be 5, 15, or 30"
                      << std::endl;
            exit(1);
        }
        };

        if (std::strcmp(k4a_color_resolution_str.c_str(), "OFF") == 0) {
            device_config.color_resolution = K4A_COLOR_RESOLUTION_OFF;
        } else if (std::strcmp(k4a_color_resolution_str.c_str(), "720P") == 0) {
            device_config.color_resolution = K4A_COLOR_RESOLUTION_720P;
        } else if (std::strcmp(k4a_color_resolution_str.c_str(), "1080P") == 0) {
            device_config.color_resolution = K4A_COLOR_RESOLUTION_1080P;
        } else if (std::strcmp(k4a_color_resolution_str.c_str(), "1440P") == 0) {
            device_config.color_resolution = K4A_COLOR_RESOLUTION_1440P;
        } else if (std::strcmp(k4a_color_resolution_str.c_str(), "1536P") == 0) {
            device_config.color_resolution = K4A_COLOR_RESOLUTION_1536P;
        } else if (std::strcmp(k4a_color_resolution_str.c_str(), "2160P") == 0) {
            device_config.color_resolution = K4A_COLOR_RESOLUTION_2160P;
        } else if (std::strcmp(k4a_color_resolution_str.c_str(), "3072P") == 0) {
            device_config.color_resolution = K4A_COLOR_RESOLUTION_3072P;
        } else {
            std::cerr << "error: color resolution must be: OFF, 720P, 1080P, "
                         "1440P, 1536P, 2160P, 3072P"
                      << std::endl;
            exit(1);
        }

        VERIFY(k4a_device_start_cameras(device, &device_config),
            "Start K4A cameras failed!");

        // Get the sensor calibration information
        VERIFY(k4a_device_get_calibration(device, device_config.depth_mode,
                   device_config.color_resolution,
                   &sensor_calibration),
            "Get depth camera calibration failed!");
    }
}

void CliConfig::printConfig() {
    //
    // Echo the configuration to the command terminal
    //
    //std::cout << "depth_mode         :" << k4a_depth_mode_str << std::endl;
    //std::cout << "color_resolution   :" << k4a_color_resolution_str << std::endl;
    //std::cout << "frames_per_second  :" << k4a_frames_per_second << std::endl;
    //std::cout << "temporal smoothing :" << temporal_smoothing << std::endl;
    //std::cout << "output file name   :" << output_json_file << std::endl;
    if (record_sensor_data) {
        //std::cout << "video file name    :" << output_sensor_file << std::endl;
    }
}
