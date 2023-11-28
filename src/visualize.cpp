#include <kinect_mocap_studio/filter_utils.hpp>
#include <kinect_mocap_studio/visualize.hpp>
#include <kinect_mocap_studio/queues.hpp>
#include <kinect_mocap_studio/utils.hpp>
#include <kinect_mocap_studio/cli.hpp>

#include <optional>
#include <thread>
#include <iostream>

#include <k4abt.h>
#include <k4a/k4a.h>
#include <k4a/k4atypes.h>
#include <k4abttypes.h>

#include "BodyTrackingHelpers.h"

#include "FloorDetector.h"
#include <Window3dWrapper.h>

#include <filter/com.hpp>

const std::chrono::duration<double, std::milli> TIME_PER_FRAME(32);

#ifndef BENCH_VIZ
#define BENCH_VIZ 1
#endif

// Taken from Azure-Kinect-Samples/body-tracking-samples/simple_3d_viewer
int64_t processKey(void* /*context*/, int key)
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
int64_t closeCallback(void* /*context*/)
{
    s_isRunning = false;
    return 1;
}

void visualizeSkeleton(Window3dWrapper& window3d, ProcessedFrame frame) {
    window3d.CleanJointsAndBones();
    int offset = 7;
    // visualize Joints
    for (int body_id = 0; body_id < frame.joints.size(); ++body_id) {
        auto body_joints = frame.joints.at(body_id);
        auto confidence_level = frame.confidence_levels.at(body_id);
        for (int i=0; i < body_joints.size(); ++i) {
            Color color = g_bodyColors[body_id + offset % g_bodyColors.size()];
            if (confidence_level.at(i) <= K4ABT_JOINT_CONFIDENCE_MEDIUM) {
                color.a = 0.1f;
            }
            add_point(window3d, body_joints.at(i), color);
        }

        // visualize bones
        for (size_t boneIdx = 0; boneIdx < g_boneList.size(); boneIdx++)
        {
            int joint1 = (int)g_boneList[boneIdx].first;
            int joint2 = (int)g_boneList[boneIdx].second;

            if (confidence_level.at(joint1) < K4ABT_JOINT_CONFIDENCE_LOW && confidence_level.at(joint2) < K4ABT_JOINT_CONFIDENCE_LOW) {
                continue;
            }

            Color color = g_bodyColors[body_id + offset % g_bodyColors.size()];

            Point<double>& joint1Position = body_joints.at(joint1);
            Point<double>& joint2Position = body_joints.at(joint2);

            if (confidence_level.at(joint1) < K4ABT_JOINT_CONFIDENCE_MEDIUM && confidence_level.at(joint2) < K4ABT_JOINT_CONFIDENCE_MEDIUM) {
                color.a = 0.1f;
            }

            add_bone(window3d, joint1Position, joint2Position, color);
        }

        if (frame.stability_properties.size() > 0) {
            // visualize stability properties
            auto [com, xcom, bos] = frame.stability_properties.at(body_id);
            add_point(window3d, com);
            add_point(window3d, xcom);

            auto [center, _] = bos.into_center_and_normal();
            add_point(window3d, center);

            linmath::vec3 a = {
                (float) bos.a.x / 1000,
                (float) bos.a.y / 1000,
                (float) bos.a.z / 1000
            };
            linmath::vec3 b = {
                (float) bos.b.x / 1000,
                (float) bos.b.y / 1000,
                (float) bos.b.z / 1000
            };
            linmath::vec3 c = {
                (float) bos.c.x / 1000,
                (float) bos.c.y / 1000,
                (float) bos.c.z / 1000
            };
            linmath::vec3 d = {
                (float) bos.d.x / 1000,
                (float) bos.d.y / 1000,
                (float) bos.d.z / 1000
            };
            window3d.SetBosRendering(true, a, b, c, d);
        }
    }

    if (frame.joints.size() == 0)
        window3d.DisableBosRendering();

}

void visualizeFloor(Window3dWrapper& window3d, std::optional<Samples::Plane> floor, nlohmann::json& frame_result_json) {
    nlohmann::json floor_result_json;
    if (floor.has_value()) {
        // For visualization purposes, make floor origin the projection of a point 1.5m in front of the camera.
        Samples::Vector cameraOrigin = { 0, 0, 0 };
        Samples::Vector cameraForward = { 0, 0, 1 };

        auto p = floor->ProjectPoint(cameraOrigin)
            + floor->ProjectVector(cameraForward) * 1.5f;

        auto n = floor->Normal;

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

void visualizePointCloud(Window3dWrapper& window3d, ProcessedFrame frame) {
    window3d.UpdatePointClouds(frame.cloudPoints);
}

void visualizeLogic(Window3dWrapper& window3d, ProcessedFrame frame, nlohmann::json& frame_result_json) {
    visualizeFloor(window3d, frame.floor, frame_result_json);
    visualizePointCloud(window3d, frame);
    visualizeSkeleton(window3d, frame);
}

void visualizeThread(k4a_calibration_t sensor_calibration) {
    ProcessedFrame frame;

    Window3dWrapper window3d;
    window3d.Create("3D Visualization", sensor_calibration);
    window3d.SetCloseCallback(closeCallback);
    window3d.SetKeyCallback(processKey);

    nlohmann::json frame_result_json;

    bool skip = false;
    while (s_isRunning) {
        auto start = std::chrono::high_resolution_clock::now();
        bool retrieved =  processed_queue.Consume(frame);
        if (retrieved and !skip) {
            skip = false;
            visualizeLogic(window3d, frame, frame_result_json);
            window3d.SetLayout3d((Visualization::Layout3d)((int)s_layoutMode));
            window3d.SetJointFrameVisualization(s_visualizeJointFrame);
            window3d.Render();
            auto stop = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> time = stop - start;
            if (time > TIME_PER_FRAME) {
                skip = true;
                std::cerr <<"Viz took to long. Will skip a frame next time." << std::endl;
            }
#ifdef BENCH_VIZ
            std::cerr << "Viz Duration: " << time.count() << "ms\n";
#endif
        } else {
            std::this_thread::yield();
        }
    }
    std::cout << "Finish visualize thread" << std::endl;
}

// // /*
// // Taken from Azure-Kinect-Samples/body-tracking-samples/simple_3d_viewer
// void visualizeResult(k4abt_frame_t bodyFrame, Window3dWrapper& window3d,
//     int depthWidth, int depthHeight, SkeletonFilterBuilder<double> builder, uint64_t timestamp)
// {
// 
//     // Obtain original capture that generates the body tracking result
//     /*
//     k4a_capture_t originalCapture = k4abt_frame_get_capture(bodyFrame);
//     k4a_image_t depthImage = k4a_capture_get_depth_image(originalCapture);
// 
//     std::vector<Color> pointCloudColors(depthWidth * depthHeight,
//         { 1.f, 1.f, 1.f, 1.f });
// 
//     // Read body index map and assign colors
//     k4a_image_t bodyIndexMap = k4abt_frame_get_body_index_map(bodyFrame);
//     const uint8_t* bodyIndexMapBuffer = k4a_image_get_buffer(bodyIndexMap);
//     for (int i = 0; i < depthWidth * depthHeight; i++) {
//         uint8_t bodyIndex = bodyIndexMapBuffer[i];
//         if (bodyIndex != K4ABT_BODY_INDEX_MAP_BACKGROUND) {
//             uint32_t bodyId = k4abt_frame_get_body_id(bodyFrame, bodyIndex);
//             pointCloudColors[i] = g_bodyColors[bodyId % g_bodyColors.size()];
//         }
//     }
//     k4a_image_release(bodyIndexMap);
//     */
// 
//     // Visualize point cloud
//     //window3d.UpdatePointClouds(depthImage, pointCloudColors);
// 
//     // Visualize the skeleton data
//     window3d.CleanJointsAndBones();
//     uint32_t numBodies = k4abt_frame_get_num_bodies(bodyFrame);
//     for (uint32_t i = 0; i < numBodies; i++) {
//         // If there is no filter for this body index generate a new one
//         if (filters.empty() or filters.size() <= numBodies) {
//             filters.push_back(builder.build());
//             std::cout << "Created a new filter." << std::endl;
//         }
// 
//         auto& filter = filters.at(i);
// 
//         k4abt_body_t body;
//         VERIFY(k4abt_frame_get_body_skeleton(bodyFrame, i, &body.skeleton),
//             "Get skeleton from body frame failed!");
//         body.id = k4abt_frame_get_body_id(bodyFrame, i);
// 
//         // Assign the correct color based on the body id
//         Color color = g_bodyColors[body.id % g_bodyColors.size()];
//         color.a = 0.4f;
//         Color lowConfidenceColor = color;
//         lowConfidenceColor.a = 0.1f;
// 
//         // Visualize joints
//         for (int joint = 0; joint < static_cast<int>(K4ABT_JOINT_COUNT); joint++) {
//             if (body.skeleton.joints[joint].confidence_level
//                 >= K4ABT_JOINT_CONFIDENCE_LOW) {
//                 const k4a_float3_t& jointPosition
//                     = body.skeleton.joints[joint].position;
//                 const k4a_quaternion_t& jointOrientation
//                     = body.skeleton.joints[joint].orientation;
// 
//                 window3d.AddJoint(
//                     jointPosition,
//                     jointOrientation,
//                     body.skeleton.joints[joint].confidence_level
//                             >= K4ABT_JOINT_CONFIDENCE_MEDIUM
//                         ? color
//                         : lowConfidenceColor);
//             }
//         }
// 
//         std::vector<Point<double>> joint_positions;
//         for (int joint = 0; joint < filter.joint_count(); ++joint) {
//             joint_positions.push_back(Point<double>(
//                 (double)body.skeleton.joints[joint].position.xyz.x,
//                 (double)body.skeleton.joints[joint].position.xyz.y,
//                 (double)body.skeleton.joints[joint].position.xyz.z));
//         }
// 
//         if (!filter.is_initialized()) {
//             filter.init(joint_positions, timestamp);
//         } else {
//             int offset = 7;
//             Color color = g_bodyColors[body.id + offset % g_bodyColors.size()];
//             color.a = 0.4f;
//             Color lowConfidenceColor = color;
//             lowConfidenceColor.a = 0.1f;
//             auto [filtered_positions, filtered_velocities] = filter.step(joint_positions, timestamp);
//             for (int joint = 0; joint < filter.joint_count(); ++joint) {
//                 auto filtered_position = filtered_positions[joint];
//                 k4a_float3_t pos;
//                 pos.v[0] = filtered_position.x;
//                 pos.v[1] = filtered_position.y;
//                 pos.v[2] = filtered_position.z;
//                 const k4a_float3_t& jointPosition = pos;
//                 const k4a_quaternion_t& jointOrientation = body.skeleton.joints[joint].orientation;
//                 // //std::cout << "Diff x: " << filtered_position.x - body.skeleton.joints[joint].position.xyz.x << std::endl;
//                 // //std::cout << "Diff y: " << filtered_position.y - body.skeleton.joints[joint].position.xyz.y << std::endl;
//                 // //std::cout << "Diff z: " << filtered_position.z - body.skeleton.joints[joint].position.xyz.z << std::endl;
// 
//                 window3d.AddJoint(
//                     jointPosition,
//                     jointOrientation,
//                     body.skeleton.joints[joint].confidence_level >= K4ABT_JOINT_CONFIDENCE_MEDIUM ? color : lowConfidenceColor);
//             }
// 
//             // Add center of mass
//             auto com = filter.calculate_com();
//             add_point(window3d, com);
// 
//             auto ankle_left = filtered_positions[ANKLE_LEFT];
//             auto ankle_right = filtered_positions[ANKLE_RIGHT];
// 
//             // Take point in the middle of both ankles
//             Point<double> mean_ankle;
//             mean_ankle.x = (ankle_left.x + ankle_right.x) / 2;
//             mean_ankle.y = (ankle_left.y + ankle_right.y) / 2;
//             mean_ankle.z = (ankle_left.z + ankle_right.z) / 2;
// 
//             // Calc euclidean norm from mean to com
//             auto ankle_com_norm = std::sqrt(
//                 std::pow(mean_ankle.x - com.x, 2) + std::pow(mean_ankle.y - com.y, 2) + std::pow(mean_ankle.z - com.z, 2));
// 
//             auto x_com = filter.calculate_x_com(ankle_com_norm);
//             add_point(window3d, x_com);
// 
//             Plane<double> bos_plane = azure_kinect_bos(filtered_positions);
//             linmath::vec3 a = {
//                 bos_plane.a.x / 1000,
//                 bos_plane.a.y / 1000,
//                 bos_plane.a.z / 1000
//             };
//             linmath::vec3 b = {
//                 bos_plane.b.x / 1000,
//                 bos_plane.b.y / 1000,
//                 bos_plane.b.z / 1000
//             };
//             linmath::vec3 c = {
//                 bos_plane.c.x / 1000,
//                 bos_plane.c.y / 1000,
//                 bos_plane.c.z / 1000
//             };
//             linmath::vec3 d = {
//                 bos_plane.d.x / 1000,
//                 bos_plane.d.y / 1000,
//                 bos_plane.d.z / 1000
//             };
//             auto [center, normal] = bos_plane.into_center_and_normal();
//             window3d.SetBosRendering(true, a, b, c, d);
//             center.x *= 1000;
//             center.y *= 1000;
//             center.z *= 1000;
//             add_point(window3d, center);
//         }
// 
//         // Visualize bones
//         for (size_t boneIdx = 0; boneIdx < g_boneList.size(); boneIdx++) {
//             k4abt_joint_id_t joint1 = g_boneList[boneIdx].first;
//             k4abt_joint_id_t joint2 = g_boneList[boneIdx].second;
// 
//             if (body.skeleton.joints[joint1].confidence_level
//                     >= K4ABT_JOINT_CONFIDENCE_LOW
//                 && body.skeleton.joints[joint2].confidence_level
//                     >= K4ABT_JOINT_CONFIDENCE_LOW) {
//                 bool confidentBone
//                     = body.skeleton.joints[joint1].confidence_level
//                         >= K4ABT_JOINT_CONFIDENCE_MEDIUM
//                     && body.skeleton.joints[joint2].confidence_level
//                         >= K4ABT_JOINT_CONFIDENCE_MEDIUM;
// 
//                 const k4a_float3_t& joint1Position
//                     = body.skeleton.joints[joint1].position;
//                 const k4a_float3_t& joint2Position
//                     = body.skeleton.joints[joint2].position;
// 
//                 window3d.AddBone(joint1Position, joint2Position,
//                     confidentBone ? color : lowConfidenceColor);
//             }
//         }
//     }
// 
//     // k4a_capture_release(originalCapture);
//     // k4a_image_release(depthImage);
// }
