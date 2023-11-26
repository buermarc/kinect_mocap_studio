﻿#include <cstdint>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <k4a/k4atypes.h>
#include <k4abttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>
#include <thread>

#include <k4a/k4a.h>
#include <k4abt.h>
#include <k4arecord/playback.h>
#include <k4arecord/record.h>

#include <eigen3/Eigen/Dense>

#include "BodyTrackingHelpers.h"
#include "FloorDetector.h"
#include "PointCloudGenerator.h"
#include "Utilities.h"
#include <Window3dWrapper.h>

#include <nlohmann/json.hpp>
#include <tclap/CmdLine.h>

#include <filter/com.hpp>
#include <filter/SkeletonFilter.hpp>


#include <kinect_mocap_studio/filter_utils.hpp>
#include <kinect_mocap_studio/visualize.hpp>
#include <kinect_mocap_studio/utils.hpp>
#include <kinect_mocap_studio/cli.hpp>
#include <kinect_mocap_studio/queues.hpp>
#include <kinect_mocap_studio/process.hpp>
#include <kinect_mocap_studio/visualize.hpp>

using Eigen::MatrixXd;

MeasurementQueue measurement_queue;
ProcessedQueue processed_queue;

#define WAIT_MS 100

boost::atomic<bool> s_isRunning (true);
boost::atomic<bool> s_visualizeJointFrame (false);
boost::atomic<int> s_layoutMode ((int) Visualization::Layout3d::OnlyMainView);

std::vector<SkeletonFilter<double>> filters;


/*
To do:
1. Use TCLAP to add some command line options
2. Add a preview of the skeleton + the room
3. Add buttons for start/stop and write
4. Add an option (pre-checked) to measure the position of the floor (averaged)
   and to add it to the json file using floor_detector_sample
5. Add an option (pre-checked) to append the trial number to the file name
6. Write a rbdl-toolkit plug in to visualize this data
*/

int main(int argc, char** argv)
{

    // To-do's
    // 1. Include TCLAP to get command line arguments.
    // 2. Add arguments for the
    //     - [required] number of seconds to record
    //     - [required] name of the json file to write
    //     - [optional] skeleton tracking smoothing parameter
    //     - [optional] name of the mkv file to write

    auto config = CliConfig(argc, argv);

    // Configure and start the device
    k4a_device_t device = NULL;
    k4a_playback_t playback_handle = NULL;

    k4a_calibration_t sensor_calibration;
    k4a_device_configuration_t device_config = K4A_DEVICE_CONFIG_INIT_DISABLE_ALL;

    config.openDeviceOrRecording(device, playback_handle, sensor_calibration, device_config);

    int depthWidth = sensor_calibration.depth_camera_calibration.resolution_width;
    int depthHeight = sensor_calibration.depth_camera_calibration.resolution_height;

    // Echo the configuration to the command terminal
    config.printConfig();

    std::thread process_thread(processThread, sensor_calibration);
    std::thread visualize_thread(visualizeThread, sensor_calibration);

    //
    // Initialize and start the body tracker
    //
    k4abt_tracker_t tracker = NULL;
    // k4abt_tracker_configuration_t tracker_config = K4ABT_TRACKER_CONFIG_DEFAULT;
    k4abt_tracker_configuration_t tracker_config = { K4ABT_SENSOR_ORIENTATION_DEFAULT, K4ABT_TRACKER_PROCESSING_MODE_GPU, 0 };
    VERIFY(k4abt_tracker_create(&sensor_calibration, tracker_config, &tracker),
        "Body tracker initialization failed!");
    k4abt_tracker_set_temporal_smoothing(tracker, config.temporal_smoothing);

    // Start the IMU
    if (!config.process_sensor_file) {
        VERIFY(k4a_device_start_imu(device), "Start K4A imu failed!");
    }
    //
    // JSON pre-amble
    //

    // Store the json pre-amble data
    nlohmann::json json_output;
    nlohmann::json frames_json = nlohmann::json::array();

    time_t now = time(0);
    char* dt = ctime(&now);
    json_output["start_time"] = dt;

    json_output["k4abt_sdk_version"] = K4ABT_VERSION_STR;

    json_output["depth_mode"] = config.k4a_depth_mode_str;
    json_output["color_resolution"] = config.k4a_color_resolution_str;
    json_output["frames_per_second"] = config.k4a_frames_per_second;
    json_output["temporal_smoothing"] = config.temporal_smoothing;

    // Store all joint names to the json
    json_output["joint_names"] = nlohmann::json::array();
    for (int i = 0; i < (int)K4ABT_JOINT_COUNT; i++) {
        json_output["joint_names"].push_back(
            g_jointNames.find((k4abt_joint_id_t)i)->second);
    }

    // Store all bone linkings to the json
    json_output["bone_list"] = nlohmann::json::array();
    for (int i = 0; i < (int)g_boneList.size(); i++) {
        json_output["bone_list"].push_back(
            { g_jointNames.find(g_boneList[i].first)->second,
                g_jointNames.find(g_boneList[i].second)->second });
    }


    k4a_record_t recording;

    if (config.record_sensor_data) {
        if (K4A_FAILED(k4a_record_create(config.output_sensor_file.c_str(), device,
                device_config, &recording))) {
            std::cerr << "error: k4a_record_create() failed, unable to create "
                      << config.output_sensor_file << std::endl;
            exit(1);
        }
        if (K4A_FAILED(k4a_record_add_imu_track(recording))) {
            std::cerr << "Error: k4a_record_add_imu_track() failed" << std::endl;
            exit(1);
        }
        if (K4A_FAILED(k4a_record_write_header(recording))) {
            std::cerr << "Error: k4a_record_write_header() failed" << std::endl;
            exit(1);
        }
    }

    //
    // Process each frame
    //
    int frame_count = 0;

    Samples::PointCloudGenerator pointCloudGenerator { sensor_calibration };

    /**
     * Skeleton Filter setup
     */
    int joint_count = 32;
    SkeletonFilterBuilder<double> skeleton_filter_builder(joint_count, 2.0);

    do {
        std::cout << "Top" << std::endl;
        k4a_capture_t sensor_capture = nullptr;

        bool capture_ready = false;



        // Extract capture
        if (config.process_sensor_file) {
            k4a_stream_result_t stream_result = k4a_playback_get_next_capture(playback_handle, &sensor_capture);
            if (stream_result == K4A_STREAM_RESULT_EOF) {
                break;
            } else if (stream_result == K4A_STREAM_RESULT_SUCCEEDED) {
                capture_ready = true;
            } else {
                std::cerr << "error: k4a_playback_get_next_capture() failed at "
                          << frame_count << std::endl;
                exit(1);
            }

            if (check_depth_image_exists(sensor_capture) == false) {
                std::cerr << "error: stream contains no depth image at " << frame_count
                          << std::endl;
                capture_ready = false;
            }

        } else {
            std::cout << "wait1" << std::endl;
            k4a_wait_result_t get_capture_result
                = k4a_device_get_capture(device, &sensor_capture, WAIT_MS);

            if (get_capture_result == K4A_WAIT_RESULT_SUCCEEDED) {
                capture_ready = true;
            } else if (get_capture_result == K4A_WAIT_RESULT_TIMEOUT) {
                printf("error: k4a_device_get_capture() timed out \n");
                // break;
            } else {
                printf("error: k4a_device_get_capture(): %d\n", get_capture_result);
                // break;
            }
        }

        // Process the data
        if (capture_ready) {

            std::cout << "wait2" << std::endl;
            k4a_wait_result_t queue_capture_result = k4abt_tracker_enqueue_capture(tracker, sensor_capture,
                WAIT_MS);
            k4a_image_t depth_image = k4a_capture_get_depth_image(sensor_capture);

            if (config.record_sensor_data) {
                k4a_result_t write_sensor_capture
                    = k4a_record_write_capture(recording, sensor_capture);

                if (K4A_FAILED(write_sensor_capture)) {
                    std::cerr << "error: k4a_record_write_capture() returned "
                              << write_sensor_capture << std::endl;
                    break;
                }
            }

            // Remember to release the sensor capture once you finish using it
            k4a_capture_release(sensor_capture);

            if (queue_capture_result == K4A_WAIT_RESULT_TIMEOUT) {
                // It should never hit timeout when K4A_WAIT_INFINITE is set.
                printf("Error! Add capture to tracker process queue timeout!\n");
                break;
            } else if (queue_capture_result == K4A_WAIT_RESULT_FAILED) {
                printf("Error! Add capture to tracker process queue failed!\n");
                break;
            }

            k4abt_frame_t body_frame = NULL;
            std::cout << "Popping tracker." << std::endl;
            k4a_wait_result_t pop_frame_result = k4abt_tracker_pop_result(tracker, &body_frame,
                WAIT_MS);
            std::cout << "Popped tracker." << std::endl;

            // Maybe we just put it onto the queue here, and everything below
            // will move somewhere else

            if (pop_frame_result == K4A_WAIT_RESULT_SUCCEEDED) {
                k4a_imu_sample_t imu_sample;


                bool imu_data_ready = false;
                if (config.process_sensor_file) {
                    k4a_stream_result_t imu_result = k4a_playback_get_next_imu_sample(playback_handle, &imu_sample);
                    if (imu_result == K4A_STREAM_RESULT_SUCCEEDED) {
                        imu_data_ready = true;
                    } else if (imu_result == K4A_STREAM_RESULT_EOF) {
                        imu_data_ready = false;
                    } else {
                        std::cerr << "error: k4a_playback_get_next_imu_sample() failed at "
                                  << frame_count << std::endl;
                        exit(1);
                    }

                } else {
                    // Maybe move this out into the front? Is there any advantage of having it in here?
                    VERIFY_WAIT(k4a_device_get_imu_sample(device, &imu_sample,
                                    WAIT_MS),
                        "Timed out waiting for IMU data");
                    imu_data_ready = true;
                }


                // BEGIN Could be its own thread
                uint32_t num_bodies = k4abt_frame_get_num_bodies(body_frame);
                uint64_t timestamp = k4abt_frame_get_device_timestamp_usec(
                    body_frame);

                nlohmann::json frame_result_json;
                frame_result_json["timestamp_usec"] = timestamp;
                frame_result_json["frame_id"] = frame_count;
                frame_result_json["num_bodies"] = num_bodies;
                frame_result_json["imu"] = nlohmann::json::array();
                frame_result_json["bodies"] = nlohmann::json::array();
                frame_result_json["floor"] = nlohmann::json::array();

                // Question:
                //  Is it faster to access bodies or to call
                //  k4abt_frame_get_body_skeleton?

                // Fetch and save the skeleton tracking data to a json object
                nlohmann::json body_result_json;
                auto joints = push_body_data_to_json(body_result_json, body_frame, num_bodies);
                frame_result_json["bodies"].push_back(body_result_json);
                // END

                // Fetch and save the imu data to a json object
                nlohmann::json imu_result_json;

                if (config.record_sensor_data) {
                    k4a_result_t write_imu_sample
                        = k4a_record_write_imu_sample(recording, imu_sample);
                    if (K4A_FAILED(write_imu_sample)) {
                        std::cerr << "error: k4a_record_write_imu_sample() returned "
                                  << write_imu_sample << std::endl;
                        // break;
                    }
                }

                if (imu_data_ready) {
                    push_imu_data_to_json(imu_result_json, imu_sample);
                    frame_result_json["imu"].push_back(imu_result_json);
                }
                // Fit a plane to the depth points that are furthest away from
                // the camera in the direction of gravity (this will fail when the
                // camera accelerates by 0.2 m/s2 in any direction)
                // This uses code from teh floor_detector example code

                /*
                if (imu_data_ready) {
                    // Update point cloud.

                    pointCloudGenerator.Update(depth_image);

                    // Get down-sampled cloud points.
                    const int downsampleStep = 2;
                    const auto& cloudPoints = pointCloudGenerator.GetCloudPoints(downsampleStep);

                    // Detect floor plane based on latest visual and inertial observations.
                    const size_t minimumFloorPointCount = 1024 / (downsampleStep * downsampleStep);
                    const auto& maybeFloorPlane = floorDetector.TryDetectFloorPlane(cloudPoints, imu_sample,
                        sensor_calibration, minimumFloorPointCount);

                    // Visualize point cloud.
                    window3d.UpdatePointClouds(depth_image);

                    // Visualize the floor plane.
                    nlohmann::json floor_result_json;
                    if (maybeFloorPlane.has_value()) {
                        // For visualization purposes, make floor origin the projection of a point 1.5m in front of the camera.
                        Samples::Vector cameraOrigin = { 0, 0, 0 };
                        Samples::Vector cameraForward = { 0, 0, 1 };

                        auto p = maybeFloorPlane->ProjectPoint(cameraOrigin)
                            + maybeFloorPlane->ProjectVector(cameraForward) * 1.5f;

                        auto n = maybeFloorPlane->Normal;

                        window3d.SetFloorRendering(true, p.X, p.Y, p.Z, n.X, n.Y, n.Z);
                        floor_result_json["point"].push_back(
                            { p.X * 1000.f, p.Y * 1000.f, p.Z * 1000.f });
                        floor_result_json["normal"].push_back({ n.X, n.Y, n.Z });
                        floor_result_json["valid"] = true;
                    } else {
                        window3d.SetFloorRendering(false, 0, 0, 0);
                        floor_result_json["point"].push_back({ 0., 0., 0. });
                        floor_result_json["normal"].push_back({ 0., 0., 0. });
                        floor_result_json["valid"] = false;
                    }
                    frame_result_json["floor"].push_back(floor_result_json);
                }
                // Vizualize the tracked result
                visualizeResult(body_frame, window3d, depthWidth, depthHeight, skeleton_filter_builder, timestamp);
                window3d.SetLayout3d((Visualization::Layout3d)((int)s_layoutMode));
                window3d.SetJointFrameVisualization(s_visualizeJointFrame);
                window3d.Render();
                */

                pointCloudGenerator.Update(depth_image);
                const auto cloudPoints = pointCloudGenerator.GetCloudPoints(2);

                std::cout << "Adding element to queue" << std::endl;
                measurement_queue.Produce(std::move(MeasuredFrame {
                    imu_sample, cloudPoints, joints, timestamp
                }));

                k4abt_frame_release(body_frame);
                k4a_image_release(depth_image);

                // Remember to release the body frame once you finish using it
                frames_json.push_back(frame_result_json);
                frame_count++;
                // auto stop = std::chrono::high_resolution_clock::now();
                // std::chrono::duration<double, std::milli> time = stop - start;
                // //std::cout << time.count() << "ms\n";
                std::cout << "End of measurement retrieval loop" << std::endl;

            } else if (pop_frame_result == K4A_WAIT_RESULT_TIMEOUT) {
                //  It should never hit timeout when K4A_WAIT_INFINITE is set.
                printf("error: timeout k4abt_tracker_pop_result()\n");
                // break;
            } else {
                printf("Pop body frame result failed!\n");
                // break;
            }
        } else {
            std::cout << "Capture was not ready." << std::endl;
        }

    } while (s_isRunning);

    printf("Finished body tracking processing!\n");

    // Write sensor data to file
    if (config.record_sensor_data) {
        k4a_record_flush(recording);
        k4a_record_close(recording);
        std::cout << "Sensor data written to " << config.output_sensor_file << std::endl;
    }
    process_thread.join();
    visualize_thread.join();

    // Write the frame_data_time_series to file
    now = time(0);
    dt = ctime(&now);
    json_output["end_time"] = dt;

    json_output["frames"] = frames_json;

    // Add the filters to the json
    json_output["filters"] = filters;
    std::ofstream output_file(config.output_json_file.c_str());
    output_file << std::setw(4) << json_output << std::endl;
    std::cout << frame_count << " Frames written to "
              << config.output_json_file << std::endl;

    //window3d.Delete();
    k4abt_tracker_shutdown(tracker);
    k4abt_tracker_destroy(tracker);

    if (config.process_sensor_file) {
        k4a_playback_close(playback_handle);
    } else {
        k4a_device_stop_cameras(device);
        k4a_device_stop_imu(device);
        k4a_device_close(device);
    }

    return 0;
}
/*
 * Handel CLI args
 * Configure atomic booleans
 * Setup data retrieval
 * Utils declares and functions e.g. VERIFY and VERIFY_WAIT
 * Questions: where does the JSON live
 * - filter json has to live in the filter thread
 * - Use Promise to return JSON
 * - Use atomic bool to indicate that we finished everything
 * - Use atomic bool to indicate that we finished everything
 *
 *  We have three threads:
 *  - Data Retrieval thread
 *  - Filter thread
 *  - Visualization thread
 *
 *  When is a thread to late?
 *  - When we took more then 33ms since the last read?
 *  - How what should we do if this happens
 *  - The skeleton filter would be fucked I guess so that should not happend
 *  - But it theory it could happen if we have to many joints
 *  - Should we spawn a thread for each body index? ??? Would yield just more questions
 *  - For the Visualization this would mean we skip a Visualization loop, basically just a continue
 *  - This means the Visualization should not influence the JSON output or the recording file
*/