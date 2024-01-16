#include "filter/com.hpp"
#include <algorithm>
#include <filter/Point.hpp>
#include <filter/Utils.hpp>
#include <filter/com.hpp>
#include <fstream>
#include <iostream>
#include <iterator>
#include <kinect_mocap_studio/filter_utils.hpp>
#include <limits>
#include <sstream>
#include <string>
#include <tclap/CmdLine.h>
#include <tuple>
#include <vector>

#include <Window3dWrapper.h>
#include <nlohmann/json.hpp>

#include <k4a/k4a.h>
#include <k4a/k4atypes.h>
#include <k4abt.h>
#include <k4abttypes.h>

bool s_isRunning = true;
bool s_visualizeJointFrame = false;
int s_layoutMode = 0;

struct Data {
    std::vector<double> timestamps;
    std::vector<Point<double>> l_ak;
    std::vector<Point<double>> r_ak;
    std::vector<Point<double>> b_ak;

    std::vector<Point<double>> l_sae;

    std::vector<Point<double>> l_hle;
    std::vector<Point<double>> l_usp;

    std::vector<Point<double>> r_hle;
    std::vector<Point<double>> r_usp;
};

struct KinectFrame {
    std::vector<Point<double>> joints;
};

struct QtmFrame {
    double timestamps;
    Point<double> l_ak;
    Point<double> r_ak;
    Point<double> b_ak;
    Point<double> l_sae;
    Point<double> l_hle;
    Point<double> l_usp;
    Point<double> r_hle;
    Point<double> r_usp;
};

Point<double> azure_kinect_origin_lab_coords(
    std::vector<Point<double>> l_ak,
    std::vector<Point<double>> r_ak,
    std::vector<Point<double>> b_ak)
{
    auto mean_l_ak = std::accumulate(l_ak.cbegin(), l_ak.cend(), Point<double>()) / l_ak.size();
    auto mean_r_ak = std::accumulate(r_ak.cbegin(), r_ak.cend(), Point<double>()) / r_ak.size();
    auto mean_b_ak = std::accumulate(b_ak.cbegin(), b_ak.cend(), Point<double>()) / b_ak.size();

    Point<double> middle_between_left_and_right = mean_l_ak + (mean_r_ak - mean_l_ak) / 2;

    auto result = (mean_r_ak - mean_l_ak).cross_product(mean_b_ak - mean_l_ak);
    return middle_between_left_and_right;
}

template <typename T>
void print_vec(std::vector<T> vector)
{
    std::for_each(vector.cbegin(), vector.cend() - 1, [](auto ele) { std::cout << ele << ", " << std::endl; });
    std::cout << vector.back() << std::endl;
}
template <typename T>
void print_vec(std::string name, std::vector<T> vector)
{
    std::cout << "@name: " << name << std::endl;
    if (vector.size() == 0) {
        std::cout << "Vector is Empty" << std::endl;
        return;
    }
    std::for_each(vector.cbegin(), vector.cend() - 1, [](auto ele) { std::cout << ele << ", " << std::endl; });
    std::cout << vector.back() << std::endl;
}

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
        std::cout << "Help" << std::endl;
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

void visualizeKinectLogic(Window3dWrapper& window3d, KinectFrame frame, Point<double> kinect_point)
{
    for (auto joint : frame.joints) {
        add_point(window3d, joint + kinect_point);
    }
}

void visualizeQtmLogic(Window3dWrapper& window3d, QtmFrame frame)
{
    add_point(window3d, frame.l_ak);
    add_point(window3d, frame.r_ak);
    add_point(window3d, frame.b_ak);

    add_point(window3d, frame.l_sae);
    add_point(window3d, frame.l_hle);
    add_point(window3d, frame.l_usp);
    add_point(window3d, frame.r_hle);
    add_point(window3d, frame.r_usp);

    linmath::vec3 a = { -1, 0.1, -1 };
    linmath::vec3 b = { 1, 0.1, -1 };
    linmath::vec3 c = { 1, 0.1, 1 };
    linmath::vec3 d = { -1, 0.1, 1 };
    window3d.SetBosRendering(true, a, b, c, d);

    add_bone(window3d, Point<double>(0, 0, 0), Point<double>(1, 0, 0), Color { 1, 0, 0, 1 });
    add_bone(window3d, Point<double>(0, 0, 0), Point<double>(0, 0, 1) * (-1), Color { 0, 1, 0, 1 });
    add_bone(window3d, Point<double>(0, 0, 0), Point<double>(0, 1, 0) * (-1), Color { 0, 0, 1, 1 });
}

std::string trimString(std::string str)
{
    const std::string whiteSpaces = " \t\n\r\f\v";
    // Remove leading whitespace
    size_t first_non_space = str.find_first_not_of(whiteSpaces);
    str.erase(0, first_non_space);
    // Remove trailing whitespace
    size_t last_non_space = str.find_last_not_of(whiteSpaces);
    str.erase(last_non_space + 1);
    return str;
}

std::vector<std::string> getNextLineAndSplitIntoTokens(std::istream& str)
{
    std::vector<std::string> result;
    std::string line;
    std::getline(str, line);

    std::stringstream lineStream(line);
    std::string cell;

    while (std::getline(lineStream, cell, '\t')) {
        result.push_back(trimString(cell));
    }
    // This checks for a trailing comma with no data after it.
    if (!lineStream && cell.empty()) {
        // If there was a trailing comma then add an empty element.
        result.push_back("");
    }
    return result;
}

class KinectRecording {
public:
    std::string json_file;
    nlohmann::json json_data;
    std::string mkv_file;
    Tensor<double, 3> joints;
    double n_frames;
    std::vector<double> timestamps;

    KinectRecording(std::string file)
    {
        auto trimmed_file(file);
        std::cout << trimmed_file << std::endl;
        if (file.find(".json") > 0) {
            trimmed_file.replace(file.find(".json"), sizeof(".json") - 1, "");

        } else if (file.find(".mkv") > 0) {
            trimmed_file.replace(file.find(".mkv"), sizeof(".mkv") - 1, "");

        } else {
            std::cerr << "Kinect recording file has a wrong extension: " << file << std::endl;
        }
        std::stringstream json_file_ss;
        std::stringstream mkv_file_ss;
        json_file_ss << trimmed_file << ".json";
        mkv_file_ss << trimmed_file << ".mkv";

        json_file = json_file_ss.str();
        mkv_file = mkv_file_ss.str();

        json_data = nlohmann::json::parse(std::ifstream(json_file));
        auto [var_joints, n_frames, timestamps, _is_null] = load_data(json_file, 32);
        this->joints = var_joints;
        this->n_frames = n_frames;
        this->timestamps = timestamps;
    }

    friend std::ostream& operator<<(std::ostream& out, KinectRecording const& recording);
};

std::ostream& operator<<(std::ostream& out, KinectRecording const& recording)
{
    out << recording.mkv_file << std::endl;
    out << recording.json_file << std::endl;

    return out;
}

struct ForcePlateData {

    Plane<double> plate;
    bool used;
    std::vector<double> timestamps;
    std::vector<Point<double>> force;
    std::vector<Point<double>> moment;
    std::vector<Point<double>> cop;
};

class QtmRecording {
public:
    QtmRecording(std::string file)
    {
        file.replace(file.find(".tsv"), sizeof(".tsv") - 1, "");
        std::stringstream marker_file_name;
        std::stringstream force_plate_file_name_f1;
        std::stringstream force_plate_file_name_f2;

        marker_file_name << file << ".tsv";
        force_plate_file_name_f1 << file << "_f_1.tsv";
        force_plate_file_name_f2 << file << "_f_2.tsv";

        this->marker_file_name = marker_file_name.str();
        this->force_plate_file_f1 = force_plate_file_name_f1.str();
        this->force_plate_file_f2 = force_plate_file_name_f2.str();
    }
    std::string marker_file_name;
    std::string force_plate_file_f1;
    std::string force_plate_file_f2;

    friend std::ostream& operator<<(std::ostream& out, QtmRecording const& recording);

    Data read_marker_file()
    {

        std::ifstream csv_file(marker_file_name);

        // Go through headers
        std::string key = "";
        do {
            auto header = getNextLineAndSplitIntoTokens(csv_file);
            if (header.size() > 0) {
                key = header.at(0);
            }
        } while (key != "Frame");

        std::vector<double> timestamps;
        std::vector<Point<double>> l_ak;
        std::vector<Point<double>> r_ak;
        std::vector<Point<double>> b_ak;

        std::vector<Point<double>> l_sae;

        std::vector<Point<double>> l_hle;
        std::vector<Point<double>> l_usp;

        std::vector<Point<double>> r_hle;
        std::vector<Point<double>> r_usp;

        while (!csv_file.eof()) {
            auto results = getNextLineAndSplitIntoTokens(csv_file);
            if (results.size() == 1) {
                break;
            }
            timestamps.push_back(std::stod(results.at(1)));
            int i;

            i = 2;
            l_ak.push_back(
                Point<double>(
                    std::stod(results.at(i + 0)) / 1000,
                    (-1) * std::stod(results.at(i + 2)) / 1000,
                    (-1) * std::stod(results.at(i + 1)) / 1000));

            i += 3;
            r_ak.push_back(
                Point<double>(
                    std::stod(results.at(i + 0)) / 1000,
                    (-1) * std::stod(results.at(i + 2)) / 1000,
                    (-1) * std::stod(results.at(i + 1)) / 1000));

            i += 3;
            b_ak.push_back(
                Point<double>(
                    std::stod(results.at(i + 0)) / 1000,
                    (-1) * std::stod(results.at(i + 2)) / 1000,
                    (-1) * std::stod(results.at(i + 1)) / 1000));

            i += 3;
            l_sae.push_back(
                Point<double>(
                    std::stod(results.at(i + 0)) / 1000,
                    (-1) * std::stod(results.at(i + 2)) / 1000,
                    (-1) * std::stod(results.at(i + 1)) / 1000));

            i += 3;
            l_hle.push_back(
                Point<double>(
                    std::stod(results.at(i + 0)) / 1000,
                    (-1) * std::stod(results.at(i + 2)) / 1000,
                    (-1) * std::stod(results.at(i + 1)) / 1000));

            i += 3;
            l_usp.push_back(
                Point<double>(
                    std::stod(results.at(i + 0)) / 1000,
                    (-1) * std::stod(results.at(i + 2)) / 1000,
                    (-1) * std::stod(results.at(i + 1)) / 1000));

            i += 3;
            r_hle.push_back(
                Point<double>(
                    std::stod(results.at(i + 0)) / 1000,
                    (-1) * std::stod(results.at(i + 2)) / 1000,
                    (-1) * std::stod(results.at(i + 1)) / 1000));

            i += 3;
            r_usp.push_back(
                Point<double>(
                    std::stod(results.at(i + 0)) / 1000,
                    (-1) * std::stod(results.at(i + 2)) / 1000,
                    (-1) * std::stod(results.at(i + 1)) / 1000));
        }

        // print_vec("timestamps", timestamps);
        // print_vec("l_ak", l_ak);
        // print_vec("r_ak", r_ak);
        // print_vec("b_ak", b_ak);
        // print_vec("l_sae", l_sae);
        // print_vec("l_hle", l_hle);
        // print_vec("l_usp", l_usp);
        // print_vec("r_hle", r_hle);
        // print_vec("r_usp", r_usp);

        return Data { timestamps, l_ak, r_ak, b_ak, l_sae, l_hle, l_usp, r_hle, r_usp };
    }

    ForcePlateData
    read_force_plate_file(std::string force_plate_file)
    {
        std::ifstream csv_file(force_plate_file);

        // Go through headers for file
        std::string key = "";
        do {
            auto header = getNextLineAndSplitIntoTokens(csv_file);
            if (header.size() > 0) {
                key = header.at(0);
            }
        } while (key != "FORCE_PLATE_NAME");

        // Get force plate placement in the lab
        double lu_x = std::stod(getNextLineAndSplitIntoTokens(csv_file).at(1));
        double lu_z = std::stod(getNextLineAndSplitIntoTokens(csv_file).at(1)) * (-1);
        double lu_y = std::stod(getNextLineAndSplitIntoTokens(csv_file).at(1)) * (-1);

        double ld_x = std::stod(getNextLineAndSplitIntoTokens(csv_file).at(1));
        double ld_z = std::stod(getNextLineAndSplitIntoTokens(csv_file).at(1)) * (-1);
        double ld_y = std::stod(getNextLineAndSplitIntoTokens(csv_file).at(1)) * (-1);

        double ru_x = std::stod(getNextLineAndSplitIntoTokens(csv_file).at(1));
        double ru_z = std::stod(getNextLineAndSplitIntoTokens(csv_file).at(1)) * (-1);
        double ru_y = std::stod(getNextLineAndSplitIntoTokens(csv_file).at(1)) * (-1);

        double rd_x = std::stod(getNextLineAndSplitIntoTokens(csv_file).at(1));
        double rd_z = std::stod(getNextLineAndSplitIntoTokens(csv_file).at(1)) * (-1);
        double rd_y = std::stod(getNextLineAndSplitIntoTokens(csv_file).at(1)) * (-1);

        auto left_up = Point<double>(lu_x, lu_y, lu_z) / 1000;
        auto left_down = Point<double>(ld_x, ld_y, ld_z) / 1000;

        auto right_up = Point<double>(ru_x, ru_y, ru_z) / 1000;
        auto right_down = Point<double>(rd_x, rd_y, rd_z) / 1000;

        auto plate = Plane<double>(left_down, right_down, right_up, left_up);

        do {
            auto header = getNextLineAndSplitIntoTokens(csv_file);
            if (header.size() > 0) {
                key = header.at(0);
            }
        } while (key != "SAMPLE");

        std::vector<double> timestamps;
        std::vector<Point<double>> force;
        std::vector<Point<double>> moment;
        std::vector<Point<double>> cop;

        while (!csv_file.eof()) {
            auto results = getNextLineAndSplitIntoTokens(csv_file);
            if (results.size() == 1) {
                break;
            }
            timestamps.push_back(std::stod(results.at(1)));
            int i;

            i = 2;
            force.push_back(
                Point<double>(
                    std::stod(results.at(i + 0)) / 1000,
                    std::stod(results.at(i + 2)) / 1000,
                    std::stod(results.at(i + 1)) / 1000));

            i += 3;
            moment.push_back(
                Point<double>(
                    std::stod(results.at(i + 0)) / 1000,
                    (-1) * std::stod(results.at(i + 2)) / 1000,
                    (-1) * std::stod(results.at(i + 1)) / 1000));

            i += 3;
            cop.push_back(
                Point<double>(
                    std::stod(results.at(i + 0)) / 1000,
                    (-1) * std::stod(results.at(i + 2)) / 1000,
                    (-1) * std::stod(results.at(i + 1)) / 1000));
        }
        auto mean = std::accumulate(force.cbegin(), force.cend(), Point<double>()) / force.size();
        bool used = (std::abs(mean.y) > 0.1);
        std::cout << "Force plate z mean: " << mean << std::endl;
        std::cout << "Force plate used: " << used << std::endl;
        return ForcePlateData { plate, used, timestamps, force, moment, cop };
    }

    std::tuple<ForcePlateData, ForcePlateData> read_force_plate_files()
    {
        auto data1 = read_force_plate_file(force_plate_file_f1);
        auto data2 = read_force_plate_file(force_plate_file_f2);

        return std::make_tuple(data1, data2);

        // print_vec("timestamp_f1", timestamp_f1);
        // print_vec("force_f1", force_f1);
        // print_vec("moment_f1", moment_f1);
        // print_vec("com_f1", com_f1);
        // print_vec("timestamp_f2", timestamp_f2);
        // print_vec("force_f2", force_f2);
        // print_vec("moment_f2", moment_f2);
        // print_vec("com_f2", com_f2);
    }
};

std::ostream& operator<<(std::ostream& out, QtmRecording const& recording)
{
    out << recording.marker_file_name << std::endl;
    out << recording.force_plate_file_f1 << std::endl;
    out << recording.force_plate_file_f2 << std::endl;

    return out;
}

class Experiment {
public:
    QtmRecording qtm_recording;
    KinectRecording kinect_recording;

    Experiment(std::string qtm_file, std::string kinect_file)
        : qtm_recording(qtm_file)
        , kinect_recording(kinect_file)
    {
    }
    friend std::ostream& operator<<(std::ostream& out, Experiment const& recording);

    void _min_max_for_one_point(
        std::vector<int>& qtm_min_events,
        std::vector<int>& qtm_max_events,
        std::vector<int>& kinect_min_events,
        std::vector<int>& kinect_max_events,
        std::vector<Point<double>> qtm,
        std::vector<Point<double>> kinect)
    {
        double min;
        double max;
        double tmp;

        int idx;

        int min_idx;
        int max_idx;

        int qtm_x_min_idx = 0;
        int qtm_x_max_idx = 0;

        int qtm_y_min_idx = 0;
        int qtm_y_max_idx = 0;

        int qtm_z_min_idx = 0;
        int qtm_z_max_idx = 0;

        int kinect_x_min_idx = 0;
        int kinect_x_max_idx = 0;

        int kinect_y_min_idx = 0;
        int kinect_y_max_idx = 0;

        int kinect_z_min_idx = 0;
        int kinect_z_max_idx = 0;

        min = std::numeric_limits<double>::max();
        max = 0;
        tmp = 0;
        idx = 0;
        min_idx = 0;
        max_idx = 0;

        for (auto point : qtm) {
            tmp = point.x;
            if (tmp > max) {
                max = tmp;
                max_idx = idx;
            }
            if (tmp < min) {
                min = tmp;
                min_idx = idx;
            }
            idx++;
        }

        qtm_min_events.push_back(min_idx);
        qtm_max_events.push_back(max_idx);

        min = std::numeric_limits<double>::max();
        max = 0;
        tmp = 0;
        idx = 0;
        min_idx = 0;
        max_idx = 0;

        for (auto point : qtm) {
            tmp = point.y;
            if (tmp > max) {
                max = tmp;
                max_idx = idx;
            }
            if (tmp < min) {
                min = tmp;
                min_idx = idx;
            }
            idx++;
        }

        qtm_min_events.push_back(min_idx);
        qtm_max_events.push_back(max_idx);

        min = std::numeric_limits<double>::max();
        max = 0;
        tmp = 0;
        idx = 0;
        min_idx = 0;
        max_idx = 0;

        for (auto point : qtm) {
            tmp = point.z;
            if (tmp > max) {
                max = tmp;
                max_idx = idx;
            }
            if (tmp < min) {
                min = tmp;
                min_idx = idx;
            }
            idx++;
        }

        qtm_min_events.push_back(min_idx);
        qtm_max_events.push_back(max_idx);

        min = std::numeric_limits<double>::max();
        max = 0;
        tmp = 0;
        idx = 0;
        min_idx = 0;
        max_idx = 0;

        for (auto point : kinect) {
            tmp = point.x;
            if (tmp > max) {
                max = tmp;
                max_idx = idx;
            }
            if (tmp < min) {
                min = tmp;
                min_idx = idx;
            }
            idx++;
        }

        kinect_min_events.push_back(min_idx);
        kinect_max_events.push_back(max_idx);

        min = std::numeric_limits<double>::max();
        max = 0;
        tmp = 0;
        idx = 0;
        min_idx = 0;
        max_idx = 0;

        for (auto point : kinect) {
            tmp = point.y;
            if (tmp > max) {
                max = tmp;
                max_idx = idx;
            }
            if (tmp < min) {
                min = tmp;
                min_idx = idx;
            }
            idx++;
        }

        kinect_min_events.push_back(min_idx);
        kinect_max_events.push_back(max_idx);

        min = std::numeric_limits<double>::max();
        max = 0;
        tmp = 0;
        idx = 0;
        min_idx = 0;
        max_idx = 0;

        for (auto point : kinect) {
            tmp = point.z;
            if (tmp > max) {
                max = tmp;
                max_idx = idx;
            }
            if (tmp < min) {
                min = tmp;
                min_idx = idx;
            }
            idx++;
        }

        kinect_min_events.push_back(min_idx);
        kinect_max_events.push_back(max_idx);
    }

    double calculate_time_offset(Data& data, Tensor<double, 3>& joints, std::vector<double> ts)
    {
        // Do this for each joint
        // Sort indicies, remove outlier -> anything above std deviation 1
        // Get mean

        std::vector<int> qtm_min_events;
        std::vector<int> qtm_max_events;
        std::vector<int> kinect_min_events;
        std::vector<int> kinect_max_events;

        std::vector<Point<double>> kinect;
        std::vector<Point<double>> qtm;

        for (int i = 0; i < ts.size(); ++i) {
            kinect.push_back(Point<double>(
                joints(i, K4ABT_JOINT_SHOULDER_LEFT, 0),
                joints(i, K4ABT_JOINT_SHOULDER_LEFT, 1),
                joints(i, K4ABT_JOINT_SHOULDER_LEFT, 2)));
        }
        _min_max_for_one_point(qtm_min_events, qtm_max_events, kinect_min_events, kinect_max_events, data.l_sae, kinect);
        kinect.clear();
        qtm.clear();

        for (int i = 0; i < data.timestamps.size(); ++i) {
            qtm.push_back(data.r_hle.at(i) + (data.l_hle.at(i) - data.r_hle.at(i)));
        }

        for (int i = 0; i < ts.size(); ++i) {
            kinect.push_back(Point<double>(
                joints(i, K4ABT_JOINT_ELBOW_LEFT, 0),
                joints(i, K4ABT_JOINT_ELBOW_LEFT, 1),
                joints(i, K4ABT_JOINT_ELBOW_LEFT, 2)));
        }
        _min_max_for_one_point(qtm_min_events, qtm_max_events, kinect_min_events, kinect_max_events, qtm, kinect);
        kinect.clear();
        qtm.clear();

        for (int i = 0; i < data.timestamps.size(); ++i) {
            qtm.push_back(data.r_usp.at(i) + (data.l_usp.at(i) - data.r_usp.at(i)));
        }

        for (int i = 0; i < ts.size(); ++i) {
            kinect.push_back(Point<double>(
                joints(i, K4ABT_JOINT_HAND_LEFT, 0),
                joints(i, K4ABT_JOINT_HAND_LEFT, 1),
                joints(i, K4ABT_JOINT_HAND_LEFT, 2)));
        }
        _min_max_for_one_point(qtm_min_events, qtm_max_events, kinect_min_events, kinect_max_events, qtm, kinect);
        kinect.clear();
        qtm.clear();

        std::vector<double> offsets;

        for (int i = 0; i < qtm_min_events.size(); ++i) {
            int qtm_idx = qtm_min_events.at(i);
            int kinect_idx = kinect_min_events.at(i);
            if (qtm_idx == 0 || kinect_idx == 0) {
                continue;
            }
            offsets.push_back(data.timestamps.at(qtm_idx) - (ts.at(kinect_idx) - ts.front()));
        }
        for (int i = 0; i < qtm_max_events.size(); ++i) {
            int qtm_idx = qtm_max_events.at(i);
            int kinect_idx = kinect_max_events.at(i);
            if (qtm_idx == 0 || kinect_idx == 0) {
                continue;
            }
            offsets.push_back(data.timestamps.at(qtm_idx) - (ts.at(kinect_idx) - ts.front()));
        }

        double offsets_mean = std::accumulate(offsets.cbegin(), offsets.cend(), 0.0) / offsets.size();
        double offsets_std = 2 * std::sqrt(std::accumulate(offsets.cbegin(), offsets.cend(), 0, [=](double a, double b) {
            return a + std::pow(b - offsets_mean, 2);
        ; })) / 3.;
        std::vector<double> offsets_filtered;

        std::copy_if(offsets.cbegin(), offsets.cend(), std::back_inserter(offsets_filtered), [=](auto element) {
            return (element > (offsets_mean - offsets_std) and (element < offsets_mean + offsets_std));
        });

        double time_offset = std::accumulate(offsets_filtered.cbegin(), offsets_filtered.cend(), 0.0) / offsets_filtered.size();
        return time_offset;

        /*
        double qtm_min_mean = std::accumulate(qtm_min_events.cbegin(), qtm_min_events.cend(), 0.0) / qtm_min_events.size();
        double qtm_max_mean = std::accumulate(qtm_max_events.cbegin(), qtm_max_events.cend(), 0.0) / qtm_max_events.size();

        double qtm_min_std = std::sqrt(std::accumulate(qtm_min_events.cbegin(), qtm_min_events.cend(), 0, [=](double a, double b) {
            return a + std::pow(b - qtm_min_mean, 2);
        ;}));
        double qtm_max_std = std::sqrt(std::accumulate(qtm_max_events.cbegin(), qtm_max_events.cend(), 0, [=](double a, double b) {
            return a + std::pow(b - qtm_max_mean, 2);
        ;}));

        std::vector<double> qtm_min_events_filtered(qtm_min_events.size());
        std::vector<double> qtm_max_events_filtered(qtm_max_events.size());

        std::copy_if(qtm_min_events.cbegin(), qtm_max_events.cend(), qtm_min_events_filtered.begin(), [=](auto element){
            return (element > (qtm_min_mean - qtm_min_std) and (element < qtm_min_mean + qtm_min_std));
        });
        std::copy_if(qtm_max_events.cbegin(), qtm_max_events.cend(), qtm_max_events_filtered.begin(), [=](auto element){
            return (element > (qtm_max_mean - qtm_max_std) and (element < qtm_max_mean + qtm_max_std));
        });

        int qtm_min_mean_filtered = (int)std::accumulate(qtm_min_events_filtered.cbegin(), qtm_min_events_filtered.cend(), 0.0) / qtm_min_events_filtered.size();
        int qtm_max_mean_filtered = (int)std::accumulate(qtm_max_events_filtered.cbegin(), qtm_max_events_filtered.cend(), 0.0) / qtm_max_events_filtered.size();


        double kinect_min_mean = std::accumulate(kinect_min_events.cbegin(), kinect_min_events.cend(), 0.0) / kinect_min_events.size();
        double kinect_max_mean = std::accumulate(kinect_max_events.cbegin(), kinect_max_events.cend(), 0.0) / kinect_max_events.size();

        double kinect_min_std = std::sqrt(std::accumulate(kinect_min_events.cbegin(), kinect_min_events.cend(), 0, [=](double a, double b) {
            return a + std::pow(b - kinect_min_mean, 2);
        ;}));
        double kinect_max_std = std::sqrt(std::accumulate(kinect_max_events.cbegin(), kinect_max_events.cend(), 0, [=](double a, double b) {
            return a + std::pow(b - kinect_max_mean, 2);
        ;}));

        std::vector<double> kinect_min_events_filtered(kinect_min_events.size());
        std::vector<double> kinect_max_events_filtered(kinect_max_events.size());

        std::copy_if(kinect_min_events.cbegin(), kinect_max_events.cend(), kinect_min_events_filtered.begin(), [=](auto element){
            return (element > (kinect_min_mean - kinect_min_std) and (element < kinect_min_mean + kinect_min_std));
        });
        std::copy_if(kinect_max_events.cbegin(), kinect_max_events.cend(), kinect_max_events_filtered.begin(), [=](auto element){
            return (element > (kinect_max_mean - kinect_max_std) and (element < kinect_max_mean + kinect_max_std));
        });

        int kinect_min_mean_filtered = (int)std::accumulate(kinect_min_events_filtered.cbegin(), kinect_min_events_filtered.cend(), 0.0) / kinect_min_events_filtered.size();
        int kinect_max_mean_filtered = (int)std::accumulate(kinect_max_events_filtered.cbegin(), kinect_max_events_filtered.cend(), 0.0) / kinect_max_events_filtered.size();

        double kinect_min_ts = ts.at(kinect_min_mean_filtered);
        double kinect_max_ts = ts.at(kinect_max_mean_filtered);

        double qtm_min_ts = data.timestamps.at(qtm_min_mean_filtered);
        double qtm_max_ts = data.timestamps.at(qtm_max_mean_filtered);

        double min_diff = kinect_min_ts - qtm_min_ts;
        double max_diff = kinect_max_ts - qtm_max_ts;
        std::cout << "min_diff: " << min_diff << std::endl;
        std::cout << "max_diff: " << max_diff << std::endl;
        std::cout << "(min_diff + max_diff) / 2: "  << (min_diff + max_diff) / 2 << std::endl;

        return (min_diff + max_diff) / 2;
        */
    }

    void visualize()
    {
        Data data = qtm_recording.read_marker_file();
        auto [force_data_f1, force_data_f2] = qtm_recording.read_force_plate_files();

        auto ts = kinect_recording.timestamps;
        auto joints = kinect_recording.joints;

        double time_offset = calculate_time_offset(data, joints, ts);
        std::cout << "Time offset: " << time_offset << std::endl;

        Window3dWrapper window3d;
        k4a_calibration_t sensor_calibration;
        sensor_calibration.depth_mode = K4A_DEPTH_MODE_NFOV_UNBINNED;
        window3d.Create("3D Visualization", sensor_calibration);
        window3d.SetCloseCallback(closeCallback);
        window3d.SetKeyCallback(processKey);

        auto camera_middle = azure_kinect_origin_lab_coords(data.l_ak, data.r_ak, data.b_ak);

        std::cout << "Kinect duration: " << ts.back() - ts.at(0) << std::endl;
        std::cout << "Qualisys duration: " << data.timestamps.back() << std::endl;

        // i is for qtm
        int i = 0;
        // q is for kinect
        int j = 0;
        if (time_offset < 0) {
            // QTM events are a bit later then the Kinect events, therefore, skip
            // a few qtm frames in the beginning
            while (data.timestamps.at(i) < time_offset) { i++; };
        } else if (time_offset > 0) {
            // Kinect events are a bit later then the Kinect events, therefore, skip
            // a few kinect frames in the beginning
            while (ts.at(j) < std::abs(time_offset)) { j++; };
        }


        auto first_ts = ts.at(0);

        QtmFrame qtm_frame;
        KinectFrame kinect_frame;

        // What ever is longer should continue
        // Never go longer over the max size
        while (i < data.timestamps.size() or j < ts.size()) {
            if (i < data.timestamps.size()) {
                qtm_frame = QtmFrame {
                    data.timestamps.at(i),
                    data.l_ak.at(i),
                    data.r_ak.at(i),
                    data.b_ak.at(i),
                    data.l_sae.at(i),
                    data.l_hle.at(i),
                    data.l_usp.at(i),
                    data.r_hle.at(i),
                    data.r_usp.at(i)
                };
            }

            if (j < ts.size()) {
                auto time = ts.at(j) - first_ts - time_offset;
                double current;
                // std::cout << "Kinect time: " << time << std::endl;
                // std::cout << "QTM time: " << current << std::endl;
                double next;
                if (i == data.timestamps.size() - 1) {
                    next = data.timestamps.at(i) + (data.timestamps.at(i) - data.timestamps.at(i - 1));
                    current = data.timestamps.at(data.timestamps.size() - 2);
                } else if (i >= data.timestamps.size()) {
                    next = ts.back();
                    current = data.timestamps.at(data.timestamps.size() - 2);
                } else {
                    current = data.timestamps.at(i);
                    next = data.timestamps.at(i + 1);
                }
                if (current <= time && time < next) {
                    std::vector<Point<double>> points;
                    for (int k = 0; k < 32; ++k) {
                        points.push_back(Point<double>(
                            joints(j, k, 0),
                            joints(j, k, 1),
                            joints(j, k, 2)));
                    }
                    kinect_frame = KinectFrame { points };
                    j++;
                }
            }
            i++;

            visualizeKinectLogic(window3d, kinect_frame, camera_middle);
            visualizeQtmLogic(window3d, qtm_frame);
            add_point(window3d, camera_middle, Color { 0, 1, 0, 1 });

            {
                // Render force plate related stuff
                add_point(window3d, force_data_f1.plate.a, Color { 0, 1, 0, 1 });
                add_point(window3d, force_data_f1.plate.b, Color { 0, 1, 0, 1 });
                add_point(window3d, force_data_f1.plate.c, Color { 0, 1, 0, 1 });
                add_point(window3d, force_data_f1.plate.d, Color { 0, 1, 0, 1 });

                add_point(window3d, force_data_f2.plate.a, Color { 0, 1, 0, 1 });
                add_point(window3d, force_data_f2.plate.b, Color { 0, 1, 0, 1 });
                add_point(window3d, force_data_f2.plate.c, Color { 0, 1, 0, 1 });
                add_point(window3d, force_data_f2.plate.d, Color { 0, 1, 0, 1 });

                int f = i * 6;
                if (f >= force_data_f1.cop.size()) {
                    f = force_data_f1.cop.size() - 1;
                }
                if (force_data_f1.used && !force_data_f2.used) {
                    add_bone(window3d, force_data_f1.cop.at(f), force_data_f1.cop.at(f) + force_data_f1.force.at(f), Color { 0, 1, 0, 1 });
                } else if (!force_data_f1.used && force_data_f2.used) {
                    add_bone(window3d, force_data_f2.cop.at(f), force_data_f2.cop.at(f) + force_data_f2.force.at(f), Color { 0, 1, 0, 1 });
                } else if (force_data_f1.used && force_data_f2.used) {
                    auto cop1 = force_data_f1.cop.at(f);
                    auto cop2 = force_data_f2.cop.at(f);

                    auto force1 = force_data_f1.force.at(f);
                    auto force2 = force_data_f2.force.at(f);
                    auto total_force = force1 + force2;
                    auto total_norm = total_force.norm();

                    auto middle = cop1 + ((cop2 - cop1) * (force2.norm() / total_norm));

                    add_bone(window3d, middle, middle + total_force, Color { 0, 1, 0, 1 });
                }
            }

            window3d.SetLayout3d((Visualization::Layout3d)((int)s_layoutMode));
            window3d.SetJointFrameVisualization(s_visualizeJointFrame);
            window3d.Render();
            window3d.CleanJointsAndBones();
            if (!s_isRunning) {
                break;
            }
        }
    }
};

std::ostream& operator<<(std::ostream& out, Experiment const& experiment)
{
    out << experiment.qtm_recording << std::endl;
    out << experiment.kinect_recording << std::endl;

    return out;
}

int main(int argc, char** argv)
{

    TCLAP::CmdLine cmd("Read tsv file from Qualisys.");

    TCLAP::ValueArg<std::string> tsv_file("q", "qtm_file",
        "TSV File from qualisys", false, "",
        "string");

    cmd.add(tsv_file);

    TCLAP::ValueArg<std::string> kinect_file("k", "kinect_file",
        "Kniect MKV or JSON File from qualisys", false, "",
        "string");
    cmd.add(kinect_file);

    cmd.parse(argc, argv);

    auto file = tsv_file.getValue();
    auto experiment = Experiment(tsv_file.getValue(), kinect_file.getValue());

    // Data data = experiment.qtm_recording.read_marker_file();
    experiment.qtm_recording.read_force_plate_files();

    std::cout << experiment << std::endl;

    experiment.visualize();
}
