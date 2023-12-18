#include <kinect_mocap_studio/utils.hpp>
#include <cstdint>
#include <k4abt.h>
#include <k4a/k4a.h>
#include <vector>
#include <map>
#include <filter/Point.hpp>

bool check_depth_image_exists(k4a_capture_t capture)
{
    k4a_image_t depth = k4a_capture_get_depth_image(capture);
    if (depth != nullptr) {
        k4a_image_release(depth);
        return true;
    } else {
        return false;
    }
}

/**
Inspired by the code in
Azure-Kinect-Samples/body-tracking-samples/offline_processor

@param imu_result_json an empty json object to populate
@param imu_sample a sample from the imu

*/
void push_imu_data_to_json(nlohmann::json& imu_result_json,
    k4a_imu_sample_t& imu_sample)
{

    imu_result_json["temperature"] = imu_sample.temperature;

    imu_result_json["acc_sample"].push_back(
        { imu_sample.acc_sample.xyz.x,
            imu_sample.acc_sample.xyz.y,
            imu_sample.acc_sample.xyz.z });

    imu_result_json["acc_timestamp_usec"] = imu_sample.acc_timestamp_usec;

    imu_result_json["gyro_sample"].push_back(
        { imu_sample.gyro_sample.xyz.x,
            imu_sample.gyro_sample.xyz.y,
            imu_sample.gyro_sample.xyz.z });

    imu_result_json["gyro_timestamp_usec"] = imu_sample.gyro_timestamp_usec;
}

/**
Inspired by the code in
Azure-Kinect-Samples/body-tracking-samples/offline_processor

@param body_result_json an empty json object to populate
@param body_frame a sample of the tracked bodies

*/
std::tuple<std::map<uint32_t, std::vector<Point<double>>>, std::map<uint32_t, std::vector<int>>>
push_body_data_to_json(nlohmann::json& body_result_json,
    k4abt_frame_t& body_frame,
    uint32_t num_bodies)
{
    std::map<uint32_t, std::vector<Point<double>>> points;
    std::map<uint32_t, std::vector<int>> confidence_levels;
    for (size_t index_body = 0; index_body < num_bodies; ++index_body) {

        k4abt_body_t body;

        // Get the skeleton
        k4abt_frame_get_body_skeleton(body_frame, index_body,
            &body.skeleton);
        // Get the body id
        body.id = k4abt_frame_get_body_id(body_frame, index_body);

        body_result_json["body_id"] = body.id;

        std::vector<Point<double>> body_joints;
        std::vector<int> confidence_level;
        body_joints.reserve(K4ABT_JOINT_COUNT);

        for (int index_joint = 0;
             index_joint < (int)K4ABT_JOINT_COUNT; ++index_joint) {
            body_result_json["joint_positions"].push_back(
                { body.skeleton.joints[index_joint].position.xyz.x,
                    body.skeleton.joints[index_joint].position.xyz.y,
                    body.skeleton.joints[index_joint].position.xyz.z });

            body_joints.push_back(Point<double>(
                (double)body.skeleton.joints[index_joint].position.v[0] / 1000,
                (double)body.skeleton.joints[index_joint].position.v[1] / 1000,
                (double)body.skeleton.joints[index_joint].position.v[2] / 1000
            ));
            body_result_json["joint_orientations"].push_back(
                { body.skeleton.joints[index_joint].orientation.wxyz.w,
                    body.skeleton.joints[index_joint].orientation.wxyz.x,
                    body.skeleton.joints[index_joint].orientation.wxyz.y,
                    body.skeleton.joints[index_joint].orientation.wxyz.z });

            body_result_json["confidence_levels"].push_back(
                body.skeleton.joints[index_joint].confidence_level);
            confidence_level.push_back(body.skeleton.joints[index_joint].confidence_level);
        }
        points[body.id] = std::move(body_joints);
        confidence_levels[body.id] = std::move(confidence_level);
    }
    return std::make_tuple(points, confidence_levels);
}
