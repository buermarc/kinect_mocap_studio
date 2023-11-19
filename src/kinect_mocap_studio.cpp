#include <cstdint>
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


#include <boost/lockfree/queue.hpp>
#include <boost/atomic.hpp>

#include <kinect_mocap_studio/filter_utils.hpp>
#include <kinect_mocap_studio/utils.hpp>
#include <kinect_mocap_studio/cli.hpp>

using Eigen::MatrixXd;

using queue = boost::lockfree::queue<k4abt_frame_t>;


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

// Global State and Key Process Function
// bool s_isRunning = true; // TODO: remove
boost::atomic<bool> s_isRunning (true);
boost::atomic<bool> s_visualizeJointFrame (false);
// bool s_visualizeJointFrame = false; // TODO: remove

// Visualization::Layout3d s_layoutMode = Visualization::Layout3d::OnlyMainView;
// Visualization::Layout3d s_layoutMode = Visualization::Layout3d::OnlyMainView;
boost::atomic<int> s_layoutMode ((int) Visualization::Layout3d::OnlyMainView);

std::vector<SkeletonFilter<double>> filters;

// Taken from Azure-Kinect-Samples/body-tracking-samples/simple_3d_viewer
// Taken from Azure-Kinect-Samples/body-tracking-samples/simple_3d_viewer
int64_t ProcessKey(void* /*context*/, int key)
{
    // https://www.glfw.org/docs/latest/group__keys.html
    switch (key) {
        // Quit
    case GLFW_KEY_ESCAPE:
        s_isRunning = false;
        break;
    case GLFW_KEY_K:
        s_layoutMode = ((int)s_layoutMode + 1) % (int)Visualization::Layout3d::Count;
        break;
    case GLFW_KEY_B:
        s_visualizeJointFrame = !s_visualizeJointFrame;
        break;
    case GLFW_KEY_H:
        CliConfig::printAppUsage();
        break;
    }
    return 1;
}

// Taken from Azure-Kinect-Samples/body-tracking-samples/simple_3d_viewer
int64_t CloseCallback(void* /*context*/)
{
    s_isRunning = false;
    return 1;
}

// Taken from Azure-Kinect-Samples/body-tracking-samples/simple_3d_viewer
void VisualizeResult(k4abt_frame_t bodyFrame, Window3dWrapper& window3d,
    int depthWidth, int depthHeight, SkeletonFilterBuilder<double> builder, uint64_t timestamp)
{

    // Obtain original capture that generates the body tracking result
    /*
    k4a_capture_t originalCapture = k4abt_frame_get_capture(bodyFrame);
    k4a_image_t depthImage = k4a_capture_get_depth_image(originalCapture);

    std::vector<Color> pointCloudColors(depthWidth * depthHeight,
        { 1.f, 1.f, 1.f, 1.f });

    // Read body index map and assign colors
    k4a_image_t bodyIndexMap = k4abt_frame_get_body_index_map(bodyFrame);
    const uint8_t* bodyIndexMapBuffer = k4a_image_get_buffer(bodyIndexMap);
    for (int i = 0; i < depthWidth * depthHeight; i++) {
        uint8_t bodyIndex = bodyIndexMapBuffer[i];
        if (bodyIndex != K4ABT_BODY_INDEX_MAP_BACKGROUND) {
            uint32_t bodyId = k4abt_frame_get_body_id(bodyFrame, bodyIndex);
            pointCloudColors[i] = g_bodyColors[bodyId % g_bodyColors.size()];
        }
    }
    k4a_image_release(bodyIndexMap);
    */

    // Visualize point cloud
    //window3d.UpdatePointClouds(depthImage, pointCloudColors);

    // Visualize the skeleton data
    window3d.CleanJointsAndBones();
    uint32_t numBodies = k4abt_frame_get_num_bodies(bodyFrame);
    for (uint32_t i = 0; i < numBodies; i++) {
        // If there is no filter for this body index generate a new one
        if (filters.empty() or filters.size() <= numBodies) {
            filters.push_back(builder.build());
            std::cout << "Created a new filter." << std::endl;
        }

        auto& filter = filters.at(i);

        k4abt_body_t body;
        VERIFY(k4abt_frame_get_body_skeleton(bodyFrame, i, &body.skeleton),
            "Get skeleton from body frame failed!");
        body.id = k4abt_frame_get_body_id(bodyFrame, i);

        // Assign the correct color based on the body id
        Color color = g_bodyColors[body.id % g_bodyColors.size()];
        color.a = 0.4f;
        Color lowConfidenceColor = color;
        lowConfidenceColor.a = 0.1f;

        // Visualize joints
        for (int joint = 0; joint < static_cast<int>(K4ABT_JOINT_COUNT); joint++) {
            if (body.skeleton.joints[joint].confidence_level
                >= K4ABT_JOINT_CONFIDENCE_LOW) {
                const k4a_float3_t& jointPosition
                    = body.skeleton.joints[joint].position;
                const k4a_quaternion_t& jointOrientation
                    = body.skeleton.joints[joint].orientation;

                window3d.AddJoint(
                    jointPosition,
                    jointOrientation,
                    body.skeleton.joints[joint].confidence_level
                            >= K4ABT_JOINT_CONFIDENCE_MEDIUM
                        ? color
                        : lowConfidenceColor);
            }
        }

        std::vector<Point<double>> joint_positions;
        for (int joint = 0; joint < filter.joint_count(); ++joint) {
            joint_positions.push_back(Point<double>(
                (double)body.skeleton.joints[joint].position.xyz.x,
                (double)body.skeleton.joints[joint].position.xyz.y,
                (double)body.skeleton.joints[joint].position.xyz.z));
        }

        if (!filter.is_initialized()) {
            filter.init(joint_positions, timestamp);
        } else {
            int offset = 7;
            Color color = g_bodyColors[body.id + offset % g_bodyColors.size()];
            color.a = 0.4f;
            Color lowConfidenceColor = color;
            lowConfidenceColor.a = 0.1f;
            auto [filtered_positions, filtered_velocities] = filter.step(joint_positions, timestamp);
            for (int joint = 0; joint < filter.joint_count(); ++joint) {
                auto filtered_position = filtered_positions[joint];
                k4a_float3_t pos;
                pos.v[0] = filtered_position.x;
                pos.v[1] = filtered_position.y;
                pos.v[2] = filtered_position.z;
                const k4a_float3_t& jointPosition = pos;
                const k4a_quaternion_t& jointOrientation = body.skeleton.joints[joint].orientation;
                // //std::cout << "Diff x: " << filtered_position.x - body.skeleton.joints[joint].position.xyz.x << std::endl;
                // //std::cout << "Diff y: " << filtered_position.y - body.skeleton.joints[joint].position.xyz.y << std::endl;
                // //std::cout << "Diff z: " << filtered_position.z - body.skeleton.joints[joint].position.xyz.z << std::endl;

                window3d.AddJoint(
                    jointPosition,
                    jointOrientation,
                    body.skeleton.joints[joint].confidence_level >= K4ABT_JOINT_CONFIDENCE_MEDIUM ? color : lowConfidenceColor);
            }

            // Add center of mass
            auto com = filter.calculate_com();
            add_point(window3d, com);

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

            auto x_com = filter.calculate_x_com(ankle_com_norm);
            add_point(window3d, x_com);

            Plane<double> bos_plane = azure_kinect_bos(filtered_positions);
            linmath::vec3 a = {
                bos_plane.a.x / 1000,
                bos_plane.a.y / 1000,
                bos_plane.a.z / 1000
            };
            linmath::vec3 b = {
                bos_plane.b.x / 1000,
                bos_plane.b.y / 1000,
                bos_plane.b.z / 1000
            };
            linmath::vec3 c = {
                bos_plane.c.x / 1000,
                bos_plane.c.y / 1000,
                bos_plane.c.z / 1000
            };
            linmath::vec3 d = {
                bos_plane.d.x / 1000,
                bos_plane.d.y / 1000,
                bos_plane.d.z / 1000
            };
            auto [center, normal] = bos_plane.into_center_and_normal();
            window3d.SetBosRendering(true, a, b, c, d);
            center.x *= 1000;
            center.y *= 1000;
            center.z *= 1000;
            add_point(window3d, center);
        }

        // Visualize bones
        for (size_t boneIdx = 0; boneIdx < g_boneList.size(); boneIdx++) {
            k4abt_joint_id_t joint1 = g_boneList[boneIdx].first;
            k4abt_joint_id_t joint2 = g_boneList[boneIdx].second;

            if (body.skeleton.joints[joint1].confidence_level
                    >= K4ABT_JOINT_CONFIDENCE_LOW
                && body.skeleton.joints[joint2].confidence_level
                    >= K4ABT_JOINT_CONFIDENCE_LOW) {
                bool confidentBone
                    = body.skeleton.joints[joint1].confidence_level
                        >= K4ABT_JOINT_CONFIDENCE_MEDIUM
                    && body.skeleton.joints[joint2].confidence_level
                        >= K4ABT_JOINT_CONFIDENCE_MEDIUM;

                const k4a_float3_t& joint1Position
                    = body.skeleton.joints[joint1].position;
                const k4a_float3_t& joint2Position
                    = body.skeleton.joints[joint2].position;

                window3d.AddBone(joint1Position, joint2Position,
                    confidentBone ? color : lowConfidenceColor);
            }
        }
    }

    // k4a_capture_release(originalCapture);
    // k4a_image_release(depthImage);
}

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

    int depthWidth
        = sensor_calibration.depth_camera_calibration.resolution_width;
    int depthHeight
        = sensor_calibration.depth_camera_calibration.resolution_height;

    // Echo the configuration to the command terminal
    config.printConfig();

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

    //
    // PointCloudGenerator for floor estimation.
    //
    Samples::PointCloudGenerator pointCloudGenerator { sensor_calibration };
    Samples::FloorDetector floorDetector;

    //
    // Visualization Window
    //

    Window3dWrapper window3d;
    window3d.Create("3D Visualization", sensor_calibration);
    window3d.SetCloseCallback(CloseCallback);
    window3d.SetKeyCallback(ProcessKey);

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

    /**
     * Skeleton Filter setup
     */
    int joint_count = 32;
    SkeletonFilterBuilder<double> skeleton_filter_builder(joint_count, 2.0);

    do {
        k4a_capture_t sensor_capture = nullptr;

        bool capture_ready = false;
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
            k4a_wait_result_t get_capture_result
                = k4a_device_get_capture(device, &sensor_capture, K4A_WAIT_INFINITE);

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

            k4a_wait_result_t queue_capture_result = k4abt_tracker_enqueue_capture(tracker, sensor_capture,
                K4A_WAIT_INFINITE);
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
            k4a_wait_result_t pop_frame_result = k4abt_tracker_pop_result(tracker, &body_frame,
                K4A_WAIT_INFINITE);

            if (pop_frame_result == K4A_WAIT_RESULT_SUCCEEDED) {
                // auto start = std::chrono::high_resolution_clock::now();

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
                push_body_data_to_json(body_result_json, body_frame, num_bodies);
                frame_result_json["bodies"].push_back(body_result_json);

                // Fetch and save the imu data to a json object
                nlohmann::json imu_result_json;
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
                    VERIFY_WAIT(k4a_device_get_imu_sample(device, &imu_sample,
                                    K4A_WAIT_INFINITE),
                        "Timed out waiting for IMU data");
                    imu_data_ready = true;
                }
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
                VisualizeResult(body_frame, window3d, depthWidth, depthHeight, skeleton_filter_builder, timestamp);
                window3d.SetLayout3d((Visualization::Layout3d)((int)s_layoutMode));
                window3d.SetJointFrameVisualization(s_visualizeJointFrame);
                window3d.Render();

                k4abt_frame_release(body_frame);
                k4a_image_release(depth_image);

                // Remember to release the body frame once you finish using it
                frames_json.push_back(frame_result_json);
                frame_count++;
                // auto stop = std::chrono::high_resolution_clock::now();
                // std::chrono::duration<double, std::milli> time = stop - start;
                // //std::cout << time.count() << "ms\n";

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

    window3d.Delete();
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
