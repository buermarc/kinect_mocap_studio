#include "WindowController3d.h"
#include "filter/com.hpp"
#include <algorithm>
#include <cmath>
#include <filter/Point.hpp>
#include <filter/Utils.hpp>
#include <filter/com.hpp>
#include <fstream>
#include <iostream>
#include <iterator>
#include <kinect_mocap_studio/filter_utils.hpp>
#include <limits>
#include <numbers>
#include <numeric>
#include <sstream>
#include <string>
#include <tclap/CmdLine.h>
#include <tuple>
#include <vector>

#include <Iir.h>

#include <libalglib/ap.h>
#include <libalglib/fasttransforms.h>

#include <Window3dWrapper.h>
#include <nlohmann/json.hpp>

#include <k4a/k4a.h>
#include <k4a/k4atypes.h>
#include <k4abt.h>
#include <k4abttypes.h>

#include <matplotlibcpp/matplotlibcpp.h>

namespace plt = matplotlibcpp;

std::vector<Point<double>> kinect_com;
std::vector<double> kinect_com_ts;
std::vector<Point<double>> qtm_cop;
std::vector<double> qtm_cop_ts;
auto MM = get_azure_kinect_com_matrix();

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
    std::vector<Point<double>> unfiltered_joints;
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

std::vector<double> smooth(std::vector<double> input, int window_size = 5)
{
    std::vector<double> result;
    for (int i = 0; i < input.size(); ++i) {
        double sum;
        double value;
        if (i < window_size) {
            sum = 0;
            for (int j = 0; j <= 2 * i; ++j) {
                sum += input.at(j);
            }
            value = sum / ((2 * i) + 1);
            result.push_back(value);
        } else if (i >= (input.size() - window_size)) {
            int amount = (input.size() - 1) - i;
            sum = 0;
            for (int j = i - amount; j <= i + amount; ++j) {
                sum += input.at(j);
            }
            result.push_back(sum / (amount * 2 + 1));
        } else {
            sum = 0;
            for (int j = i - window_size; j <= i + window_size; ++j) {
                sum += input.at(j);
            }
            value = sum / (window_size * 2 + 1);
            result.push_back(value);
        }
    }
    return result;
}

std::tuple<Point<double>, MatrixXd> translation_and_rotation(
    std::vector<Point<double>> l_ak,
    std::vector<Point<double>> r_ak,
    std::vector<Point<double>> b_ak)
{
    auto mean_l_ak = std::accumulate(l_ak.cbegin(), l_ak.cend(), Point<double>()) / l_ak.size();
    auto mean_r_ak = std::accumulate(r_ak.cbegin(), r_ak.cend(), Point<double>()) / r_ak.size();
    auto mean_b_ak = std::accumulate(b_ak.cbegin(), b_ak.cend(), Point<double>()) / b_ak.size();

    Point<double> translation = mean_l_ak + (mean_r_ak - mean_l_ak) / 2;

    Point<double> x, y, z;

    MatrixXd rotation_matrix(3, 3);

    x = mean_r_ak - mean_l_ak;
    z = translation - mean_b_ak;

    // Shift both down N cm
    y = x.cross_product(z);
    y = y * (-1);
    y = y.normalized();
    y = y * 0.025;

    mean_l_ak = mean_l_ak + y;
    mean_r_ak = mean_r_ak + y;
    mean_b_ak = mean_b_ak + y;

    translation = mean_l_ak + (mean_r_ak - mean_l_ak) / 2;

    // Recalculate x and z
    x = mean_r_ak - mean_l_ak;
    z = translation - mean_b_ak;

    double deg = (M_PI / 180.) * 6.;
    double z_norm = z.norm();
    auto w = x.cross_product(z);
    w = w * (-1);

    auto rotated_z = ((z * (std::cos(deg) / z_norm)) + (w * (std::sin(deg) / w.norm()))) * z_norm;
    auto rotated_y = (x.cross_product(rotated_z));

    rotated_y = rotated_y * (-1);

    x = x.normalized();
    y = rotated_y.normalized();
    z = rotated_z.normalized();

    // x = x * (-1);
    // z = z*(-1);

    std::cout << "x : " << x << std::endl;
    std::cout << "y : " << y << std::endl;
    std::cout << "z : " << z << std::endl;

    // Add 6 degrees
    std::cout << "rotated z : " << rotated_z << std::endl;

    rotation_matrix(0, 0) = x.x;
    rotation_matrix(1, 0) = x.y;
    rotation_matrix(2, 0) = x.z;

    rotation_matrix(0, 1) = y.x;
    rotation_matrix(1, 1) = y.y;
    rotation_matrix(2, 1) = y.z;

    rotation_matrix(0, 2) = z.x;
    rotation_matrix(1, 2) = z.y;
    rotation_matrix(2, 2) = z.z;

    return std::make_tuple(translation, rotation_matrix);
}

template <typename T>
void print_vec(std::vector<T> vector)
{
    std::for_each(vector.cbegin(), vector.cend() - 1, [](auto ele) { std::cout << ele << ", "; });
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
    std::for_each(vector.cbegin(), vector.cend() - 1, [](auto ele) { std::cout << ele << ", "; });
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

void visualizeKinectLogic(Window3dWrapper& window3d, KinectFrame frame, Point<double> translation, MatrixXd rotation)
{
    Color pink = Color { 1, 0, 0.8, 1 };
    Color yellow = Color { 1, 0.9, 0, 1 };
    for (auto joint : frame.unfiltered_joints) {
        ;
        add_qtm_point(window3d, joint, pink);
    }

    if (frame.unfiltered_joints.size() == 32) {
        add_qtm_point(window3d, com_helper(frame.unfiltered_joints, MM), yellow);
    }

    for (auto joint : frame.joints) {
        add_qtm_point(window3d, joint);
    }

    if (frame.joints.size() == 32) {
        add_qtm_point(window3d, com_helper(frame.joints, MM), Color { 0, 1, 0, 1 });
    }
}

void visualizeQtmLogic(Window3dWrapper& window3d, QtmFrame frame)
{
    Color light_blue { 0, 0, 1, 0.15 };
    Color blue { 0, 0, 1, 1 };
    add_qtm_point(window3d, frame.l_ak);
    add_qtm_point(window3d, frame.r_ak);
    add_qtm_point(window3d, frame.b_ak);

    add_qtm_point(window3d, frame.l_sae, blue);

    add_qtm_point(window3d, frame.l_hle, light_blue);
    add_qtm_point(window3d, frame.r_hle, light_blue);

    add_qtm_point(window3d, frame.l_usp, light_blue);
    add_qtm_point(window3d, frame.r_usp, light_blue);

    add_qtm_point(window3d, frame.r_hle * 0.5 + frame.l_hle * 0.5, blue);
    add_qtm_point(window3d, frame.r_hle * 0.5 + frame.l_hle * 0.5, blue);

    add_qtm_point(window3d, frame.r_usp * 0.5 + frame.l_usp * 0.5, blue);
    add_qtm_point(window3d, frame.r_usp * 0.5 + frame.l_usp * 0.5, blue);

    linmath::vec3 a = { -1, 0.1, -1 };
    linmath::vec3 b = { 1, 0.1, -1 };
    linmath::vec3 c = { 1, 0.1, 1 };
    linmath::vec3 d = { -1, 0.1, 1 };
    window3d.SetBosRendering(true, a, b, c, d);

    add_bone(window3d, Point<double>(0, 0, 0), Point<double>(1, 0, 0), Color { 1, 0, 0, 1 });
    add_bone(window3d, Point<double>(0, 0, 0), Point<double>(0, 1, 0), Color { 0, 1, 0, 1 });
    add_bone(window3d, Point<double>(0, 0, 0), Point<double>(0, 0, 1), Color { 0, 0, 1, 1 });
    /*
    add_bone(window3d, Point<double>(0, 0, 0), Point<double>(1, 0, 0), Color { 1, 0, 0, 1 });
    add_bone(window3d, Point<double>(0, 0, 0), Point<double>(0, 0, 1) * (-1), Color { 0, 1, 0, 1 });
    add_bone(window3d, Point<double>(0, 0, 0), Point<double>(0, 1, 0) * (-1), Color { 0, 0, 1, 1 });
    */
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
    Tensor<double, 3> unfiltered_joints;
    Tensor<double, 3> joints;
    double n_frames;
    std::vector<double> timestamps;

    KinectRecording() { }

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
        auto [un_var_joints, fn_frames, ftimestamps, _f_is_null] = load_data(json_file, 32);
        auto [var_joints, n_frames, timestamps, _is_null] = load_filtered_data(json_file, 32);

        this->unfiltered_joints = un_var_joints;
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
    QtmRecording() { }
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
                    std::stod(results.at(i + 1)) / 1000,
                    std::stod(results.at(i + 2)) / 1000));

            i += 3;
            r_ak.push_back(
                Point<double>(
                    std::stod(results.at(i + 0)) / 1000,
                    std::stod(results.at(i + 1)) / 1000,
                    std::stod(results.at(i + 2)) / 1000));

            i += 3;
            b_ak.push_back(
                Point<double>(
                    std::stod(results.at(i + 0)) / 1000,
                    std::stod(results.at(i + 1)) / 1000,
                    std::stod(results.at(i + 2)) / 1000));

            i += 3;
            l_sae.push_back(
                Point<double>(
                    std::stod(results.at(i + 0)) / 1000,
                    std::stod(results.at(i + 1)) / 1000,
                    std::stod(results.at(i + 2)) / 1000));

            i += 3;
            l_hle.push_back(
                Point<double>(
                    std::stod(results.at(i + 0)) / 1000,
                    std::stod(results.at(i + 1)) / 1000,
                    std::stod(results.at(i + 2)) / 1000));

            i += 3;
            l_usp.push_back(
                Point<double>(
                    std::stod(results.at(i + 0)) / 1000,
                    std::stod(results.at(i + 1)) / 1000,
                    std::stod(results.at(i + 2)) / 1000));

            i += 3;
            r_hle.push_back(
                Point<double>(
                    std::stod(results.at(i + 0)) / 1000,
                    std::stod(results.at(i + 1)) / 1000,
                    std::stod(results.at(i + 2)) / 1000));

            i += 3;
            r_usp.push_back(
                Point<double>(
                    std::stod(results.at(i + 0)) / 1000,
                    std::stod(results.at(i + 1)) / 1000,
                    std::stod(results.at(i + 2)) / 1000));
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
        double lu_y = std::stod(getNextLineAndSplitIntoTokens(csv_file).at(1));
        double lu_z = std::stod(getNextLineAndSplitIntoTokens(csv_file).at(1));

        double ld_x = std::stod(getNextLineAndSplitIntoTokens(csv_file).at(1));
        double ld_y = std::stod(getNextLineAndSplitIntoTokens(csv_file).at(1));
        double ld_z = std::stod(getNextLineAndSplitIntoTokens(csv_file).at(1));

        double ru_x = std::stod(getNextLineAndSplitIntoTokens(csv_file).at(1));
        double ru_y = std::stod(getNextLineAndSplitIntoTokens(csv_file).at(1));
        double ru_z = std::stod(getNextLineAndSplitIntoTokens(csv_file).at(1));

        double rd_x = std::stod(getNextLineAndSplitIntoTokens(csv_file).at(1));
        double rd_y = std::stod(getNextLineAndSplitIntoTokens(csv_file).at(1));
        double rd_z = std::stod(getNextLineAndSplitIntoTokens(csv_file).at(1));

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
                    std::stod(results.at(i + 1)) / 1000,
                    (-1) * std::stod(results.at(i + 2)) / 1000));

            i += 3;
            moment.push_back(
                Point<double>(
                    std::stod(results.at(i + 0)) / 1000,
                    std::stod(results.at(i + 1)) / 1000,
                    std::stod(results.at(i + 2)) / 1000));

            i += 3;
            cop.push_back(
                Point<double>(
                    std::stod(results.at(i + 0)) / 1000,
                    std::stod(results.at(i + 1)) / 1000,
                    std::stod(results.at(i + 2)) / 1000));
        }
        auto mean = std::accumulate(force.cbegin(), force.cend(), Point<double>()) / force.size();
        bool used = (std::abs(mean.z) > 0.1);
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
    Experiment() { }
    QtmRecording qtm_recording;
    KinectRecording kinect_recording;
    bool hard_offset;
    double offset;

    Experiment(std::string experiment_json)
    {
        std::ifstream file(experiment_json);
        json data = json::parse(file);
        std::string qtm_file = data["qtm_file"];
        std::string kinect_file = data["kinect_file"];

        if (data.contains("offset")) {
            hard_offset = true;
            offset = data["offset"];
        } else {
            hard_offset = false;
            offset = 0;
        }

        qtm_recording = QtmRecording(qtm_file);
        kinect_recording = KinectRecording(kinect_file);
    }

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

    double cross_correlation_lag(Data& data, Tensor<double, 3>& joints, std::vector<double> kinect_ts, double initial_offset, bool plot = false)
    {
        // downsample to 15hz
        auto qtm_ts = data.timestamps;

        std::vector<double> qtm_hle_y;
        // std::transform(data.l_hle.cbegin(), data.l_hle.cend(), std::back_inserter(qtm_hle_y), [](auto point) {return point.y;});

        for (int i = 0; i < data.l_usp.size(); ++i) {
            auto l = data.l_usp.at(i) * .5;
            auto r = data.r_usp.at(i) * .5;
            double value = (l + r).z;
            qtm_hle_y.push_back(value);
        }

        std::vector<double> kinect_hle_y;
        for (int i = 0; i < joints.dimension(0); ++i) {
            kinect_hle_y.push_back(joints(i, K4ABT_JOINT_WRIST_LEFT, 2));
        }

        std::vector<double> downsampled_qtm_hle_y;

        double frame_duration = 1. / 15.;

        std::cout << "initial_offset: " << initial_offset << std::endl;
        int qtm_begin_idx = 0;
        if (initial_offset < 0) {
            qtm_begin_idx = (std::abs(initial_offset) / frame_duration);
        }
        std::cout << "qtm_begin_idx: " << qtm_begin_idx << std::endl;

        for (int i = qtm_begin_idx, down_i = qtm_begin_idx; i < qtm_ts.size(); ++i) {
            auto time = qtm_ts.at(i);
            if (time >= frame_duration * down_i && time < frame_duration * (down_i + 1)) {
                downsampled_qtm_hle_y.push_back(qtm_hle_y.at(i));
                down_i++;
            }
        }

        int kinect_begin_idx = 0;
        if (initial_offset > 0) {
            for (int k = 0; k < kinect_ts.size(); ++k) {
                auto time = kinect_ts.at(k) - kinect_ts.front();
                if (time >= initial_offset) {
                    kinect_begin_idx = k;
                    break;
                }
            }
            int kinect_begin_idx = (initial_offset / frame_duration);
            std::cout << "kinect_begin_idx: " << kinect_begin_idx << std::endl;
        }

        std::vector<double> downsampled_kinect_hle_y;

        for (int i = kinect_begin_idx, down_i = kinect_begin_idx; i < kinect_ts.size(); ++i) {
            auto time = kinect_ts.at(i) - kinect_ts.front();
            double value;
            if (time >= frame_duration * down_i) {
                if (time != frame_duration * down_i) {
                    auto before_time = kinect_ts.at(i - 1) - kinect_ts.front();
                    auto before_value = kinect_hle_y.at(i - 1);
                    auto current_value = kinect_hle_y.at(i);
                    value = before_value + ((current_value - before_value) * ((frame_duration * down_i - before_time) / (time - before_time)));
                } else {
                    value = kinect_hle_y.at(i);
                }
                downsampled_kinect_hle_y.push_back(value);
                down_i++;
            }
        }

        // downsampled_qtm_hle_y = {0., 0., 0., 1., 2., 1., 0., 0., 0., 0., 0.};
        //  downsampled_kinect_hle_y = {0., 0., 0., 0., 0., 1., 3., 1., 0.};
        // downsampled_kinect_hle_y = {0., 1., 3., 1., 0.};

        auto orig_downsampled_kinect_hle_y = downsampled_kinect_hle_y;
        auto orig_downsampled_qtm_hle_y = downsampled_qtm_hle_y;

        // downsampled_kinect_hle_y = smooth(downsampled_kinect_hle_y, 7);
        // downsampled_qtm_hle_y = smooth(downsampled_qtm_hle_y, 7);

        /*
        double mean_qtm = std::accumulate(downsampled_qtm_hle_y.cbegin(), downsampled_qtm_hle_y.cend(), 0.0) / downsampled_qtm_hle_y.size();
        double mean_kinect = std::accumulate(downsampled_kinect_hle_y.cbegin(), downsampled_kinect_hle_y.cend(), 0.0) / downsampled_kinect_hle_y.size();

        double diff = mean_qtm - mean_kinect;
        std::cout << "diff: " << diff << std::endl;

        std::transform(downsampled_kinect_hle_y.cbegin(), downsampled_kinect_hle_y.cend(), downsampled_kinect_hle_y.begin(), [=](auto element) {return element + diff;});

        */
        std::cout << "qtm down size: " << downsampled_qtm_hle_y.size() << std::endl;
        std::cout << "kinect down size: " << downsampled_kinect_hle_y.size() << std::endl;

        /*
        downsampled_qtm_hle_y.clear();
        for (int i = 0; i < 150; ++i) {
            downsampled_qtm_hle_y.push_back(i);
        }

        downsampled_kinect_hle_y.clear();
        for (int i = 10; i < 160; ++i) {
            downsampled_kinect_hle_y.push_back(i);
        }
        */

        alglib_impl::ae_state state;
        ae_state_init(&state);

        alglib_impl::ae_vector qtm;
        memset(&qtm, 0, sizeof(qtm));

        alglib_impl::ae_vector kinect;
        memset(&kinect, 0, sizeof(kinect));

        alglib_impl::ae_vector result;
        memset(&result, 0, sizeof(result));

        ae_vector_init(&qtm, 0, alglib_impl::DT_REAL, &state, ae_true);
        ae_vector_init(&kinect, 0, alglib_impl::DT_REAL, &state, ae_true);
        ae_vector_init(&result, 0, alglib_impl::DT_REAL, &state, ae_true);

        ae_vector_set_length(&qtm, downsampled_qtm_hle_y.size(), &state);
        std::cout << "signal = [";
        for (int i = 0; i < downsampled_qtm_hle_y.size(); ++i) {
            qtm.ptr.p_double[i] = downsampled_qtm_hle_y.at(i);
            std::cout << downsampled_qtm_hle_y.at(i) << ",";
        }
        std::cout << "]" << std::endl;

        ae_vector_set_length(&kinect, downsampled_kinect_hle_y.size(), &state);
        std::cout << "sample = [";
        for (int i = 0; i < downsampled_kinect_hle_y.size(); ++i) {
            kinect.ptr.p_double[i] = downsampled_kinect_hle_y.at(i);
            std::cout << downsampled_kinect_hle_y.at(i) << ",";
        }
        std::cout << "]" << std::endl;

        ae_vector_set_length(&result, downsampled_kinect_hle_y.size() + downsampled_qtm_hle_y.size(), &state);
        corrr1d(&qtm, downsampled_qtm_hle_y.size(), &kinect, downsampled_kinect_hle_y.size(), &result, &state);

        double tmp = 0.0;
        int arg_max = 0;
        for (int i = 0; i < (downsampled_kinect_hle_y.size() + downsampled_qtm_hle_y.size()) - 1; ++i) {
            if (tmp < result.ptr.p_double[i]) {
                tmp = result.ptr.p_double[i];
                arg_max = i;
                std::cout << "Max : " << tmp;
            }
        }

        std::cout << "Initial arg max: " << arg_max << std::endl;
        if (arg_max >= downsampled_qtm_hle_y.size()) {
            arg_max = arg_max - (downsampled_qtm_hle_y.size() + downsampled_kinect_hle_y.size() - 1);
        }
        std::cout << "N" << downsampled_qtm_hle_y.size() << std::endl;
        std::cout << "M" << downsampled_kinect_hle_y.size() << std::endl;
        std::cout << "Arg max: " << arg_max << std::endl;

        /*
        std::cout << "Other way" << std::endl;
        double signalData[14] = { 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9};
        //                                    0  1  2  3  4  5
        double patternData[10] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
        alglib::real_1d_array signal;
        alglib::real_1d_array pattern;
        alglib::real_1d_array corrResult;

        signal.setcontent(14, signalData);
        pattern.setcontent(10, patternData);

        corrr1d(signal, 14, pattern, 10, corrResult);

        for (int i = 0; i < 24; ++i) {
            std::cout << corrResult[i]<< std::endl;
        }
        */

        if (plot) {
            std::vector<double> qtm_timestamp;
            for (int i = 0; i < downsampled_qtm_hle_y.size(); ++i) {
                qtm_timestamp.push_back((1. / 15.) * i);
            }
            std::vector<double> kinect_timestamp;
            for (int i = 0; i < downsampled_kinect_hle_y.size(); ++i) {
                kinect_timestamp.push_back((1. / 15.) * i);
            }

            auto front =  kinect_ts.front();
            std::transform(kinect_ts.cbegin(), kinect_ts.cend(), kinect_ts.begin(), [front](auto e) {return e - front;});

            plt::title("Smooth QTM");
            plt::named_plot("normal qtm", data.timestamps, qtm_hle_y);
            plt::named_plot("downsampled qtm", qtm_timestamp, orig_downsampled_qtm_hle_y);
            plt::named_plot("normal kinect", kinect_ts, kinect_hle_y);
            plt::named_plot("downsampled kinect", kinect_timestamp, orig_downsampled_kinect_hle_y);
            plt::legend();
            plt::show(true);
            plt::cla();

            plt::title("Downsampled Wrist Z");
            plt::named_plot("QTM", qtm_timestamp, downsampled_qtm_hle_y);
            plt::named_plot("Kinect", kinect_timestamp, downsampled_kinect_hle_y);
            plt::legend();
            plt::show(true);
            plt::cla();

            /*
            std::vector<double> shifted_downsampled_kinect_hle_y;
            std::vector<double> orig_shifted_downsampled_kinect_hle_y;
            for (int i = arg_max; i < downsampled_kinect_hle_y.size(); ++i) {
                shifted_downsampled_kinect_hle_y.push_back(downsampled_kinect_hle_y.at(i));
                orig_shifted_downsampled_kinect_hle_y.push_back(orig_downsampled_kinect_hle_y.at(i));
            }
            */

            int before_size = qtm_timestamp.size();

            std::vector<double> shifted_kinect_timestamp;
            std::cout << "downsampled size :" << downsampled_kinect_hle_y.size() << std::endl;
            for (int i = arg_max; i < (arg_max + (int)downsampled_kinect_hle_y.size()); ++i) {
                shifted_kinect_timestamp.push_back((1. / 15.) * (i));
            }

            plt::title("Shifted Downsampled Wrist Z");
            plt::named_plot("QTM", qtm_timestamp, orig_downsampled_qtm_hle_y);
            plt::named_plot("Shifted Kinect", shifted_kinect_timestamp, downsampled_kinect_hle_y);
            plt::named_plot("Normal Kinect", kinect_timestamp, downsampled_kinect_hle_y, "r--");
            plt::legend();
            plt::show(true);
            plt::cla();

            /*
            plt::title("Smooth Shifted Downsampled Wrist Z");
            plt::named_plot("Shifted QTM", qtm_timestamp, smooth(downsampled_qtm_hle_y));
            plt::named_plot("Shifted Kinect", kinect_timestamp, shifted_downsampled_kinect_hle_y);
            plt::legend();
            plt::show(true);
            plt::cla();
            */
        }

        return initial_offset + (-(1. / 15.) * arg_max);
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

    Tensor<double, 3> transform_and_rotate(Tensor<double, 3> joints_in_kinect_system, Point<double> translation, MatrixXd rotation)
    {

        int frames = joints_in_kinect_system.dimension(0);
        int joint_count = joints_in_kinect_system.dimension(1);
        Tensor<double, 3> joints(frames, joint_count, 3);

        Point<double> tmp;
        for (int i = 0; i < frames; ++i) {
            for (int j = 0; j < joint_count; ++j) {
                tmp.x = joints_in_kinect_system(i, j, 0);
                tmp.y = joints_in_kinect_system(i, j, 1);
                tmp.z = joints_in_kinect_system(i, j, 2);

                // auto result = tmp.mat_mul(rotation);
                // auto result = tmp + translation;
                // auto result = tmp.mat_mul(rotation);
                auto result = tmp.mat_mul(rotation) + translation;

                joints(i, j, 0) = result.x;
                joints(i, j, 1) = result.y;
                joints(i, j, 2) = result.z;
            }
        }
        return joints;
    }

    void visualize()
    {
        Data data = qtm_recording.read_marker_file();
        auto [force_data_f1, force_data_f2] = qtm_recording.read_force_plate_files();

        auto ts = kinect_recording.timestamps;
        auto joints_in_kinect_system = kinect_recording.joints;
        auto unfiltered_joints_in_kinect_system = kinect_recording.unfiltered_joints;

        Window3dWrapper window3d;
        k4a_calibration_t sensor_calibration;
        sensor_calibration.depth_mode = K4A_DEPTH_MODE_NFOV_UNBINNED;
        window3d.Create("3D Visualization", sensor_calibration);
        window3d.SetCloseCallback(closeCallback);
        window3d.SetKeyCallback(processKey);
        window3d.SetTopViewPoint();
        window3d.Scroll(10);

        auto [translation, rotation] = translation_and_rotation(data.l_ak, data.r_ak, data.b_ak);

        auto joints = transform_and_rotate(joints_in_kinect_system, translation, rotation);
        auto unfiltered_joints = transform_and_rotate(unfiltered_joints_in_kinect_system, translation, rotation);

        double time_offset = 0;
        if (this->hard_offset) {
            time_offset = this->offset;
        }
        time_offset = cross_correlation_lag(data, joints, ts, this->offset, false);
        std::cout << "Time offset: " << time_offset << std::endl;

        std::cout << "Kinect duration: " << ts.back() - ts.at(0) << std::endl;
        std::cout << "Qualisys duration: " << data.timestamps.back() << std::endl;

        // i is for qtm
        int i = 0;
        // q is for kinect
        int j = 0;
        if (time_offset < 0) {
            // QTM events are a bit later then the Kinect events, therefore, skip
            // a few qtm frames in the beginning
            while (data.timestamps.at(i) < time_offset) {
                i++;
            };
        } else if (time_offset > 0) {
            // Kinect events are a bit later then the Kinect events, therefore, skip
            // a few kinect frames in the beginning
            while ((ts.at(j) - ts.front()) < std::abs(time_offset)) {
                j++;
            };
        }
        std::cout << "Begin: i=" << i << ", j=" << j << std::endl;

        auto first_ts = ts.at(0);

        QtmFrame qtm_frame;
        KinectFrame kinect_frame;

        std::vector<double> y1;
        std::vector<double> y2;
        std::vector<double> y3;
        std::vector<double> plot_ts;

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
                    std::vector<Point<double>> unfiltered_points;
                    for (int k = 0; k < 32; ++k) {
                        points.push_back(Point<double>(
                            joints(j, k, 0),
                            joints(j, k, 1),
                            joints(j, k, 2)));
                        unfiltered_points.push_back(Point<double>(
                            unfiltered_joints(j, k, 0),
                            unfiltered_joints(j, k, 1),
                            unfiltered_joints(j, k, 2)));
                    }
                    kinect_frame = KinectFrame { points, unfiltered_points };
                    auto o = std::min(i, (int)data.timestamps.size() - 1);
                    auto m = data.l_usp.at(o) * 0.5 + data.r_usp.at(o) * 0.5;
                    y1.push_back(m.z);
                    y2.push_back(points.at(K4ABT_JOINT_WRIST_LEFT).z);
                    y3.push_back(unfiltered_points.at(K4ABT_JOINT_WRIST_LEFT).z);
                    auto com = com_helper(points, MM);
                    com.z = 0;
                    kinect_com.push_back(com);
                    kinect_com_ts.push_back(ts.at(j) - ts.front() - time_offset);
                    Color yellow = Color { 1, 0.9, 0, 1 };
                    add_qtm_point(window3d, com, yellow);
                    plot_ts.push_back(current);
                    j++;
                }
            }
            i++;

            visualizeKinectLogic(window3d, kinect_frame, translation, rotation);
            visualizeQtmLogic(window3d, qtm_frame);
            add_qtm_point(window3d, translation, Color { 0, 1, 0, 1 });

            {
                // Render force plate related stuff
                add_qtm_point(window3d, force_data_f1.plate.a, Color { 0, 1, 0, 1 });
                add_qtm_point(window3d, force_data_f1.plate.b, Color { 0, 1, 0, 1 });
                add_qtm_point(window3d, force_data_f1.plate.c, Color { 0, 1, 0, 1 });
                add_qtm_point(window3d, force_data_f1.plate.d, Color { 0, 1, 0, 1 });

                add_qtm_point(window3d, force_data_f2.plate.a, Color { 0, 1, 0, 1 });
                add_qtm_point(window3d, force_data_f2.plate.b, Color { 0, 1, 0, 1 });
                add_qtm_point(window3d, force_data_f2.plate.c, Color { 0, 1, 0, 1 });
                add_qtm_point(window3d, force_data_f2.plate.d, Color { 0, 1, 0, 1 });

                int f = i * 6;
                if (f >= force_data_f1.cop.size()) {
                    f = force_data_f1.cop.size() - 1;
                }
                Point<double> center, cop;
                if (force_data_f1.used && !force_data_f2.used) {
                    center = force_data_f1.cop.at(f);
                    cop = force_data_f1.cop.at(f) + force_data_f1.force.at(f);
                } else if (!force_data_f1.used && force_data_f2.used) {
                    center = force_data_f2.cop.at(f);
                    cop = force_data_f2.cop.at(f) + force_data_f2.force.at(f);
                } else if (force_data_f1.used && force_data_f2.used) {
                    auto cop1 = force_data_f1.cop.at(f);
                    auto cop2 = force_data_f2.cop.at(f);

                    auto force1 = force_data_f1.force.at(f).z;
                    auto force2 = force_data_f2.force.at(f).z;
                    auto total_force_z = force1 + force2;
                    auto total_force = force_data_f1.force.at(f) + force_data_f2.force.at(f);

                    auto middle = cop1 + ((cop2 - cop1) * (force2 / total_force_z));
                    center = middle;
                    cop = middle + total_force;

                }
                add_qtm_bone(window3d, center, cop, Color { 0, 1, 0, 1 });
                center.z = 0;
                if (i < data.timestamps.size()) {
                    qtm_cop.push_back(center);
                    qtm_cop_ts.push_back(data.timestamps.at(i));
                }
                Color blue = Color { 0, 0, 1, 1 };
                add_qtm_point(window3d, center, blue);
            }

            window3d.SetLayout3d((Visualization::Layout3d)((int)s_layoutMode));
            window3d.SetJointFrameVisualization(s_visualizeJointFrame);
            // window3d.Render();
            window3d.CleanJointsAndBones();
            if (!s_isRunning) {
                break;
            }
        }

        plt::title("Left Wrist Z");
        plt::named_plot("qtm", plot_ts, y1);
        plt::named_plot("kinect", plot_ts, y2);
        plt::named_plot("unfiltered kinect", plot_ts, y3);
        plt::legend();
        plt::show(true);
        plt::cla();

        std::vector<double> kx, ky, qx, qy, mean_kx, mean_ky, mean_qx, mean_qy;
        std::transform(kinect_com.cbegin(), kinect_com.cend(), std::back_inserter(kx), [](auto point) {return point.x;});
        std::transform(kinect_com.cbegin(), kinect_com.cend(), std::back_inserter(ky), [](auto point) {return point.y;});

        mean_kx.push_back(std::accumulate(kx.cbegin(), kx.cend(), 0.0) / kx.size());
        mean_ky.push_back(std::accumulate(ky.cbegin(), ky.cend(), 0.0) / ky.size());

        std::transform(qtm_cop.cbegin(), qtm_cop.cend(), std::back_inserter(qx), [](auto point) {return point.x;});
        std::transform(qtm_cop.cbegin(), qtm_cop.cend(), std::back_inserter(qy), [](auto point) {return point.y;});

        mean_qx.push_back(std::accumulate(qx.cbegin(), qx.cend(), 0.0) / qx.size());
        mean_qy.push_back(std::accumulate(qy.cbegin(), qy.cend(), 0.0) / qy.size());

        plt::title("Projected Kinect CoM & QTM CoP");
        plt::scatter(kx, ky, 1.0, {{"label", "Kinect"}});
        plt::scatter(qx, qy, 1.0, {{"label", "QTM"}});
        plt::scatter(mean_kx, mean_ky, 45.0, {{"label", "Mean Kinect"}, {"marker", "X"}});
        plt::scatter(mean_qx, mean_qy, 45.0, {{"label", "Mean QTM"}, {"marker", "X"}});
        plt::xlabel("X axis [meter]");
        plt::ylabel("Y axis [meter]");
        plt::legend();
        plt::show(true);
        plt::cla();

        Iir::Butterworth::LowPass<3> bfqx, bfqy, bfkx, bfky;
	const float samplingrate = 15; // Hz
	const float cutoff_frequency = 0.5; // Hz

        std::vector<double> dkx, dky, dqx, dqy;

        double frame_duration = 1. / 15.;
        for (int i = 0, down_i = 0; i < qtm_cop_ts.size(); ++i) {
            auto time = qtm_cop_ts.at(i);
            if (time >= frame_duration * down_i && time < frame_duration * (down_i + 1)) {
                dqx.push_back(qx.at(i));
                dqy.push_back(qy.at(i));
                down_i++;
            }
        }

        for (int i = 0, down_i = 0; i < kinect_com_ts.size(); ++i) {
            auto time = kinect_com_ts.at(i) - kinect_com_ts.front();
            double value;
            if (time >= frame_duration * down_i) {
                if (time != frame_duration * down_i) {
                    auto before_time = kinect_com_ts.at(i - 1) - kinect_com_ts.front();
                    auto before_value = kx.at(i - 1);
                    auto current_value = kx.at(i);
                    value = before_value + ((current_value - before_value) * ((frame_duration * down_i - before_time) / (time - before_time)));
                } else {
                    value = kx.at(i);
                }
                dkx.push_back(value);
                down_i++;
            }
        }

        for (int i = 0, down_i = 0; i < kinect_com_ts.size(); ++i) {
            auto time = kinect_com_ts.at(i) - kinect_com_ts.front();
            double value;
            if (time >= frame_duration * down_i) {
                if (time != frame_duration * down_i) {
                    auto before_time = kinect_com_ts.at(i - 1) - kinect_com_ts.front();
                    auto before_value = ky.at(i - 1);
                    auto current_value = ky.at(i);
                    value = before_value + ((current_value - before_value) * ((frame_duration * down_i - before_time) / (time - before_time)));
                } else {
                    value = ky.at(i);
                }
                dky.push_back(value);
                down_i++;
            }
        }

        std::vector<double> downsampled_kinect_ts, downsampled_qtm_ts;

        for (int i = 0; i < dkx.size(); ++i) {
            downsampled_kinect_ts.push_back((frame_duration * i) - time_offset);
        }
        for (int i = 0; i < dqx.size(); ++i) {
            downsampled_qtm_ts.push_back(frame_duration * i);
        }


	// calc the coefficients
	bfqx.setup(samplingrate, cutoff_frequency);
	bfqy.setup(samplingrate, cutoff_frequency);
	bfkx.setup(samplingrate, cutoff_frequency);
	bfky.setup(samplingrate, cutoff_frequency);

        double kx_mean = std::accumulate(dkx.cbegin(), dkx.cend(), 0.0) / dkx.size();
        double ky_mean = std::accumulate(dky.cbegin(), dky.cend(), 0.0) / dky.size();
        double qx_mean = std::accumulate(dqx.cbegin(), dqx.cend(), 0.0) / dqx.size();
        double qy_mean = std::accumulate(dqy.cbegin(), dqy.cend(), 0.0) / dqy.size();

        std::vector<double> fqx, fqy, fkx, fky;
        for (int i = 0; i < downsampled_qtm_ts.size(); ++i) {
            fqx.push_back(qx_mean+bfqx.filter(dqx.at(i)-qx_mean));
            fqy.push_back(qy_mean+bfqy.filter(dqy.at(i)-qy_mean));
        }
        for (int i = 0; i < downsampled_kinect_ts.size(); ++i) {
            fkx.push_back(kx_mean+bfkx.filter(dkx.at(i)-kx_mean));
            fky.push_back(ky_mean+bfky.filter(dky.at(i)-ky_mean));
        }

        plt::title("CoM and CoP Movement X");
        plt::named_plot("QTM", qtm_cop_ts, qx);
        plt::named_plot("Kinect", kinect_com_ts, kx);
        plt::named_plot("Butterworth QTM", downsampled_qtm_ts, fqx);
        plt::named_plot("Butterworth Kinect", downsampled_kinect_ts, fkx);
        plt::xlabel("Time");
        plt::ylabel("X axis [meter]");
        plt::legend();
        plt::show(true);
        plt::cla();

        plt::title("CoM and CoP Movement Y");
        plt::named_plot("QTM", qtm_cop_ts, qy);
        plt::named_plot("Kinect", kinect_com_ts, ky);
        plt::named_plot("Butterworth QTM", downsampled_qtm_ts, fqy);
        plt::named_plot("Butterworth Kinect", downsampled_kinect_ts, fky);
        plt::xlabel("Time");
        plt::ylabel("Y axis [meter]");
        plt::legend();
        plt::show(true);
        plt::cla();


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

    TCLAP::ValueArg<std::string> experiment_json("e", "experiment_json",
        "Experiment JSON containing info about experiment", false, "",
        "string");

    cmd.add(experiment_json);

    cmd.parse(argc, argv);

    auto file = tsv_file.getValue();
    Experiment experiment;
    if (file != "") {
        experiment = Experiment(tsv_file.getValue(), kinect_file.getValue());
    } else {
        experiment = Experiment(experiment_json.getValue());
    }

    // Data data = experiment.qtm_recording.read_marker_file();
    experiment.qtm_recording.read_force_plate_files();

    std::cout << experiment << std::endl;

    experiment.visualize();
}
