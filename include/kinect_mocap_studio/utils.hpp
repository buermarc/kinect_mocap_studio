#pragma once
#include <nlohmann/json.hpp>
#include <k4a/k4atypes.h>
#include <k4abttypes.h>
#include <filter/Point.hpp>

#define VERIFY(result, error)                                  \
    if (result != K4A_RESULT_SUCCEEDED) {                      \
        printf("%s \n - (File: %s, Function: %s, Line: %d)\n", \
            error, __FILE__, __FUNCTION__, __LINE__);          \
        exit(1);                                               \
    }

#define VERIFY_WAIT(result, error)                             \
    if (result != K4A_WAIT_RESULT_SUCCEEDED) {                 \
        printf("%s \n - (File: %s, Function: %s, Line: %d)\n", \
            error, __FILE__, __FUNCTION__, __LINE__);          \
        exit(1);                                               \
    }

bool check_depth_image_exists(k4a_capture_t capture);

void push_imu_data_to_json(nlohmann::json& imu_result_json, k4a_imu_sample_t& imu_sample);

std::vector<std::vector<Point<double>>> push_body_data_to_json(nlohmann::json& body_result_json, k4abt_frame_t& body_frame, uint32_t num_bodies);
