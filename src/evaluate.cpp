#include <kinect_mocap_studio/filter_utils.hpp>
#include<string>
#include<tuple>
#include<algorithm>
#include<vector>
#include<fstream>
#include<iostream>
#include<sstream>
#include<tclap/CmdLine.h>
#include<filter/Point.hpp>

#include <Window3dWrapper.h>

#include <k4abt.h>
#include <k4a/k4a.h>
#include <k4a/k4atypes.h>
#include <k4abttypes.h>

class KinectRecording {
    public:
    std::string json_file;
    std::string mkv_file;

    KinectRecording(std::string file) {
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
    }
    friend std::ostream& operator<<(std::ostream& out, KinectRecording const& recording);
};

std::ostream& operator<<(std::ostream& out, KinectRecording const& recording)
{
    out << recording.mkv_file << std::endl;
    out << recording.json_file << std::endl;

    return out;
}

class QtmRecording {
    public:
    QtmRecording(std::string file) {
        file.replace(file.find(".tsv"), sizeof(".tsv") - 1, "");
        std::stringstream marker_file_name;
        std::stringstream force_plate_file_name_f1;
        std::stringstream force_plate_file_name_f2;

        marker_file_name << file <<  ".tsv";
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

    Experiment(std::string qtm_file, std::string kinect_file) : qtm_recording(qtm_file), kinect_recording(kinect_file) {}
    friend std::ostream& operator<<(std::ostream& out, Experiment const& recording);
};

std::ostream& operator<<(std::ostream& out, Experiment const& experiment)
{
    out << experiment.qtm_recording << std::endl;
    out << experiment.kinect_recording << std::endl;

    return out;
}

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

struct Frame {
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

bool s_isRunning = true;
bool s_visualizeJointFrame = false;
int s_layoutMode = 0;

Point<double> azure_kinect_origin_lab_coords(
    std::vector<Point<double>> l_ak,
    std::vector<Point<double>> r_ak,
    std::vector<Point<double>> b_ak
) {
    auto mean_l_ak = std::accumulate(l_ak.cbegin(), l_ak.cend(), Point<double>()) / l_ak.size();
    auto mean_r_ak = std::accumulate(r_ak.cbegin(), r_ak.cend(), Point<double>()) / r_ak.size();
    auto mean_b_ak = std::accumulate(b_ak.cbegin(), b_ak.cend(), Point<double>()) / b_ak.size();

   Point<double> middle_between_left_and_right = mean_l_ak + (mean_r_ak - mean_l_ak) / 2;

    auto result = (mean_r_ak - mean_l_ak).cross_product(mean_b_ak - mean_l_ak);
    return middle_between_left_and_right;
}

template<typename T>
void print_vec(std::vector<T> vector) {
    std::for_each(vector.cbegin(), vector.cend()-1, [](auto ele){ std::cout << ele << ", " << std::endl;});
    std::cout << vector.back() << std::endl;
}
template<typename T>
void print_vec(std::string name, std::vector<T> vector) {
    std::cout << "@name: " << name << std::endl;
    std::for_each(vector.cbegin(), vector.cend()-1, [](auto ele){ std::cout << ele << ", " << std::endl;});
    std::cout << vector.back() << std::endl;
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

Data read_marker_file(std::string file) {

    std::ifstream csv_file(file);

    // Go through headers
    std::string key = "";
    do {
        auto header = getNextLineAndSplitIntoTokens(csv_file);
        if (header.size() > 0) {
            key = header.at(0);
        }
        for (auto element : header) {
            std::cout << element << " ";
        }
        std::cout << std::endl;;
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
            for (auto e : results) {
                std::cout << e << std::endl;
            }
            break;
        }
        timestamps.push_back(std::stod(results.at(1)));
        int i;

        i = 2;
        l_ak.push_back(
            Point<double>(
                std::stod(results.at(i + 0)),
                (-1) * std::stod(results.at(i + 2)),
                (-1) * std::stod(results.at(i + 1))
            )
        );

        i += 3;
        r_ak.push_back(
            Point<double>(
                std::stod(results.at(i + 0)),
                (-1) * std::stod(results.at(i + 2)),
                (-1) * std::stod(results.at(i + 1))
            )
        );

        i += 3;
        b_ak.push_back(
            Point<double>(
                std::stod(results.at(i + 0)),
                (-1) * std::stod(results.at(i + 2)),
                (-1) * std::stod(results.at(i + 1))
            )
        );

        i += 3;
        l_sae.push_back(
            Point<double>(
                std::stod(results.at(i + 0)),
                (-1) * std::stod(results.at(i + 2)),
                (-1) * std::stod(results.at(i + 1))
            )
        );

        i += 3;
        l_hle.push_back(
            Point<double>(
                std::stod(results.at(i + 0)),
                (-1) * std::stod(results.at(i + 2)),
                (-1) * std::stod(results.at(i + 1))
            )
        );

        i += 3;
        l_usp.push_back(
            Point<double>(
                std::stod(results.at(i + 0)),
                (-1) * std::stod(results.at(i + 2)),
                (-1) * std::stod(results.at(i + 1))
            )
        );

        i += 3;
        r_hle.push_back(
            Point<double>(
                std::stod(results.at(i + 0)),
                (-1) * std::stod(results.at(i + 2)),
                (-1) * std::stod(results.at(i + 1))
            )
        );

        i += 3;
        r_usp.push_back(
            Point<double>(
                std::stod(results.at(i + 0)),
                (-1) * std::stod(results.at(i + 2)),
                (-1) * std::stod(results.at(i + 1))
            )
        );
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

    return Data { timestamps , l_ak , r_ak , b_ak , l_sae, l_hle , l_usp , r_hle , r_usp };
}

std::tuple<
    std::vector<double>,
    std::vector<Point<double>>,
    std::vector<Point<double>>,
    std::vector<Point<double>>
>
read_force_plate_file(std::string force_plate_file) {
    std::ifstream csv_file(force_plate_file);

    // Go through headers for file
    std::string key = "";
    do {
        auto header = getNextLineAndSplitIntoTokens(csv_file);
        if (header.size() > 0) {
            key = header.at(0);
        }
        for (auto element : header) {
            std::cout << element << " ";
        }
        std::cout << std::endl;;
    } while (key != "SAMPLE");



    std::vector<double> timestamps;
    std::vector<Point<double>> force;
    std::vector<Point<double>> moment;
    std::vector<Point<double>> cop;

    while (!csv_file.eof()) {
        auto results = getNextLineAndSplitIntoTokens(csv_file);
        if (results.size() == 1) {
            for (auto e : results) {
                std::cout << e << std::endl;
            }
            break;
        }
        timestamps.push_back(std::stod(results.at(1)));
        int i;

        i = 2;
        force.push_back(
            Point<double>(
                std::stod(results.at(i + 0)),
                std::stod(results.at(i + 1)),
                std::stod(results.at(i + 2))
            )
        );

        i += 3;
        moment.push_back(
            Point<double>(
                std::stod(results.at(i + 0)),
                std::stod(results.at(i + 1)),
                std::stod(results.at(i + 2))
            )
        );

        i += 3;
        cop.push_back(
            Point<double>(
                std::stod(results.at(i + 0)),
                std::stod(results.at(i + 1)),
                std::stod(results.at(i + 2))
            )
        );
    }
    return std::make_tuple(timestamps, force, moment, cop);
}

void read_force_plate_files(std::string force_plate_file_f1, std::string force_plate_file_f2) {
    auto [timestamp_f1, force_f1, moment_f1, com_f1] = read_force_plate_file(force_plate_file_f1);
    auto [timestamp_f2, force_f2, moment_f2, com_f2] = read_force_plate_file(force_plate_file_f2);

    // print_vec("timestamp_f1", timestamp_f1);
    // print_vec("force_f1", force_f1);
    // print_vec("moment_f1", moment_f1);
    // print_vec("com_f1", com_f1);
    // print_vec("timestamp_f2", timestamp_f2);
    // print_vec("force_f2", force_f2);
    // print_vec("moment_f2", moment_f2);
    // print_vec("com_f2", com_f2);
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

void visualizeLogic(Window3dWrapper& window3d, Frame frame) {
    window3d.CleanJointsAndBones();
    add_point(window3d, frame.l_ak / 1000);
    add_point(window3d, frame.r_ak / 1000);
    add_point(window3d, frame.b_ak / 1000);

    add_point(window3d, frame.l_sae / 1000);
    add_point(window3d, frame.l_hle / 1000);
    add_point(window3d, frame.l_usp / 1000);
    add_point(window3d, frame.r_hle / 1000);
    add_point(window3d, frame.r_usp / 1000);

    linmath::vec3 a = { -1, 0.1, -1 };
    linmath::vec3 b = { 1, 0.1, -1};
    linmath::vec3 c = { 1, 0.1, 1};
    linmath::vec3 d = { -1, 0.1, 1 };
    window3d.SetBosRendering(true, a, b, c, d);

    add_bone(window3d, Point<double>(0, 0, 0), Point<double>(1, 0, 0), Color {1, 0, 0, 1});
    add_bone(window3d, Point<double>(0, 0, 0), Point<double>(0, 0, 1) * (-1), Color {0, 1, 0, 1});
    add_bone(window3d, Point<double>(0, 0, 0), Point<double>(0, 1, 0) * (-1), Color {0, 0, 1, 1});
}

void visualize(Data data) {
    Window3dWrapper window3d;
    k4a_calibration_t sensor_calibration;
    sensor_calibration.depth_mode = K4A_DEPTH_MODE_NFOV_UNBINNED;
    window3d.Create("3D Visualization", sensor_calibration);
    window3d.SetCloseCallback(closeCallback);
    window3d.SetKeyCallback(processKey);

    auto camera_middle =  azure_kinect_origin_lab_coords(data.l_ak, data.r_ak, data.b_ak) / 1000;

    for (int i = 0; i < data.timestamps.size(); ++i) {
        Frame frame {
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

        visualizeLogic(window3d, frame);
        add_point(window3d, camera_middle, Color {0, 1, 0, 1});
        window3d.SetLayout3d((Visualization::Layout3d)((int)s_layoutMode));
        window3d.SetJointFrameVisualization(s_visualizeJointFrame);
        window3d.Render();
        if (!s_isRunning) {
            break;
        }
    }

}

int main(int argc, char** argv) {

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

    Data data = read_marker_file(experiment.qtm_recording.marker_file_name);
    read_force_plate_files(experiment.qtm_recording.force_plate_file_f1, experiment.qtm_recording.force_plate_file_f2);

    std::cout << experiment << std::endl;

    visualize(data);
}
