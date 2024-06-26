#include <kinect_mocap_studio/cli.hpp>
#include <kinect_mocap_studio/filter_utils.hpp>
#include <kinect_mocap_studio/plotting.hpp>
#include <kinect_mocap_studio/queues.hpp>
#include <kinect_mocap_studio/utils.hpp>

#include <future>
#include <iostream>
#include <memory>
#include <optional>
#include <thread>

#include <k4a/k4a.h>
#include <k4a/k4atypes.h>
#include <k4abt.h>
#include <k4abttypes.h>

#include "BodyTrackingHelpers.h"

#include "FloorDetector.h"
#include "GLFW/glfw3.h"
#include "GLPL/rendering/IDrawable.h"
#include "GLPL/rendering/ShaderSet.h"
#include "linmath.h"
#include <Window3dWrapper.h>

#include <GLPL/plot/plot.h>
#include <GLPL/window/window.h>
#include <filter/com.hpp>
#include <glm/gtx/string_cast.hpp>

void plotThread()
{
    while (!s_glfwInitialized) {
        std::this_thread::yield();
    }
    PlottingFrame frame;
    std::map<uint32_t, std::vector<Point<double>>> tmp_joints;

    int windowWidth = 1600;
    int windowHeight = 800;
    std::shared_ptr<GLPL::IWindow> window = std::shared_ptr<GLPL::IWindow>(new GLPL::Window(windowWidth, windowHeight, false, false));
    std::shared_ptr<GLPL::Window> window2 = std::dynamic_pointer_cast<GLPL::Window>(window);

    std::vector<float> xVec1a_vel = {};
    std::vector<float> yVec1a_vel = {};
    std::vector<float> xVec1b_vel = {};
    std::vector<float> yVec1b_vel = {};
    std::vector<float> xVec1c_vel = {};
    std::vector<float> yVec1c_vel = {};

    xVec1a_vel.reserve(2000);
    yVec1a_vel.reserve(2000);
    xVec1b_vel.reserve(2000);
    yVec1b_vel.reserve(2000);
    xVec1c_vel.reserve(2000);
    yVec1c_vel.reserve(2000);

    std::vector<float> xVec1a_fd = {};
    std::vector<float> yVec1a_fd = {};
    std::vector<float> xVec1b_fd = {};
    std::vector<float> yVec1b_fd = {};
    std::vector<float> xVec1c_fd = {};
    std::vector<float> yVec1c_fd = {};

    xVec1a_fd.reserve(2000);
    yVec1a_fd.reserve(2000);
    xVec1b_fd.reserve(2000);
    yVec1b_fd.reserve(2000);
    xVec1c_fd.reserve(2000);
    yVec1c_fd.reserve(2000);

    std::vector<float> xx = {};
    std::vector<float> xy = {};
    std::vector<float> xz = {};
    std::vector<float> comdotx = {};
    std::vector<float> comdoty = {};
    std::vector<float> comdotz = {};
    xx.reserve(2000);
    xy.reserve(2000);
    xz.reserve(2000);

    std::shared_ptr<GLPL::Plot> myplot = std::make_shared<GLPL::Plot>(0.0, 0.0, 1.0, 1.0, window2->getParentDimensions(), 1, 2);
    std::shared_ptr<GLPL::IDrawable> myPlotPt = std::dynamic_pointer_cast<GLPL::IDrawable>(myplot);
    window2->addPlot(myPlotPt);

    std::shared_ptr<GLPL::Axes2D> axesPt = std::dynamic_pointer_cast<GLPL::Axes2D>(myplot->getAxes(0));
    axesPt->setAxesBoxOn(false);
    axesPt->setButtonState("Grid", false);
    axesPt->setXLabel("Frame");
    axesPt->setYLabel("Velocity (m/s)");
    axesPt->setTitle("Hand Right Velocities");
    axesPt->setYLabelRotation(GLPL::SIDEWAYS_RIGHT);
    axesPt->setButtonState("X Axes Limits Scaling", false);
    axesPt->setButtonState("Y Axes Limits Scaling", true);
    axesPt->showLegend(true);

    std::shared_ptr<GLPL::Axes2D> axesPt2 = myplot->add2DAxes();
    axesPt2->setAxesBoxOn(false);
    axesPt2->setButtonState("Grid", false);
    axesPt2->setXLabel("Frame");
    axesPt2->setYLabel("Velocity (m/s)");
    axesPt2->setTitle("CoM Velocity");
    axesPt2->setYLabelRotation(GLPL::SIDEWAYS_RIGHT);
    axesPt2->setButtonState("X Axes Limits Scaling", false);
    axesPt2->setButtonState("Y Axes Limits Scaling", true);
    axesPt2->showLegend(true);
    /*
    std::vector<std::shared_ptr<GLPL::Axes2D>> axes;
    for (int i = 1; i < 32; ++i) {
        std::shared_ptr<GLPL::Axes2D> axesPt = myplot->add2DAxes();
        axes.push_back(axesPt);
        axesPt->setAxesBoxOn(false);
        axesPt->setButtonState("Grid", false);
        axesPt->setXLabel("Time (s)");
        axesPt->setYLabel("Displacement (m)");
        std::stringstream stream;
        stream << "Joint " << i;
        axesPt->setTitle(stream.str());
        axesPt->setYLabelRotation(GLPL::SIDEWAYS_RIGHT);
        axesPt->setButtonState("X Axes Limits Scaling", false);
        axesPt->setButtonState("Y Axes Limits Scaling", true);
        axesPt->showLegend(true);
    }
    */

    // X Axis
    std::shared_ptr<GLPL::ILine2D> line1a_vel = axesPt->addLine(&xVec1a_vel, &yVec1a_vel, GLPL::SINGLE_LINE, LC_BLUE, 0.8, "X Axis - Filter Velocity");
    line1a_vel->setLineWidth(4);
    std::shared_ptr<GLPL::Line2D2Vecs> line1a_vel_cast = std::dynamic_pointer_cast<GLPL::Line2D2Vecs>(line1a_vel);

    // Y Axis
    std::shared_ptr<GLPL::ILine2D> line1b_vel = axesPt->addLine(&xVec1b_vel, &yVec1b_vel, GLPL::SINGLE_LINE, LC_GREEN, 0.8, "Y Axis - Filter Velocity");
    line1b_vel->setLineWidth(4);
    std::shared_ptr<GLPL::Line2D2Vecs> line1b_vel_cast = std::dynamic_pointer_cast<GLPL::Line2D2Vecs>(line1b_vel);

    // Z Axis
    std::shared_ptr<GLPL::ILine2D> line1c_vel = axesPt->addLine(&xVec1c_vel, &yVec1c_vel, GLPL::SINGLE_LINE, LC_MAGENTA, 0.8, "Z Axis - Filter Velocity");
    line1c_vel->setLineWidth(4);
    std::shared_ptr<GLPL::Line2D2Vecs> line1c_vel_cast = std::dynamic_pointer_cast<GLPL::Line2D2Vecs>(line1c_vel);

    // X Axis
    std::shared_ptr<GLPL::ILine2D> line1a_fd = axesPt->addLine(&xVec1a_fd, &yVec1a_fd, GLPL::SINGLE_LINE, LC_BLUE, 0.1, "X Axis - Filter Finite Diff");
    std::shared_ptr<GLPL::Line2D2Vecs> line1a_fd_cast = std::dynamic_pointer_cast<GLPL::Line2D2Vecs>(line1a_fd);

    // Y Axis
    std::shared_ptr<GLPL::ILine2D> line1b_fd = axesPt->addLine(&xVec1b_fd, &yVec1b_fd, GLPL::SINGLE_LINE, LC_GREEN, 0.1, "Y Axis - Filter Finite Diff");
    std::shared_ptr<GLPL::Line2D2Vecs> line1b_fd_cast = std::dynamic_pointer_cast<GLPL::Line2D2Vecs>(line1b_fd);

    // Z Axis
    std::shared_ptr<GLPL::ILine2D> line1c_fd = axesPt->addLine(&xVec1c_fd, &yVec1c_fd, GLPL::SINGLE_LINE, LC_MAGENTA, 0.1, "Z Axis - Filter Finite Diff");
    std::shared_ptr<GLPL::Line2D2Vecs> line1c_fd_cast = std::dynamic_pointer_cast<GLPL::Line2D2Vecs>(line1c_fd);

    // X Axis
    std::shared_ptr<GLPL::ILine2D> com_dot_x = axesPt2->addLine(&xx, &comdotx, GLPL::SINGLE_LINE, LC_BLUE, 0.5, "X Axis - Filter Velocity");
    std::shared_ptr<GLPL::Line2D2Vecs> com_dot_x_cast = std::dynamic_pointer_cast<GLPL::Line2D2Vecs>(com_dot_x);

    // Y Axis
    std::shared_ptr<GLPL::ILine2D> com_dot_y = axesPt2->addLine(&xy, &comdoty, GLPL::SINGLE_LINE, LC_GREEN, 0.5, "Y Axis - Filter Velocity");
    std::shared_ptr<GLPL::Line2D2Vecs> com_dot_y_cast = std::dynamic_pointer_cast<GLPL::Line2D2Vecs>(com_dot_y);

    // Z Axis
    std::shared_ptr<GLPL::ILine2D> com_dot_z = axesPt2->addLine(&xz, &comdotz, GLPL::SINGLE_LINE, LC_MAGENTA, 0.5, "Z Axis - Filter Velocity");
    std::shared_ptr<GLPL::Line2D2Vecs> com_dot_z_cast = std::dynamic_pointer_cast<GLPL::Line2D2Vecs>(com_dot_z);

    float yVal1a, yVal1b, yVec1c = 0;
    float i = 0;

    bool skip = false;
    bool first = true;

    window2->preLoopDraw(true);
    myplot->Draw();
    window2->postLoopDraw();

    while (s_isRunning) {
        bool retrieved = plotting_queue.Consume(frame);
        if (retrieved) {
            if (skip) {
                std::cout << "Plotting is skipping" << std::endl;
                skip = false;
                continue;
            }
            if (first) {
                tmp_joints = std::move(frame.unfiltered_joints);
                first = false;
                continue;
            }

            window2->preLoopDraw(true);

            line1a_vel_cast->dataPtX->push_back(i);
            line1b_vel_cast->dataPtX->push_back(i);
            line1c_vel_cast->dataPtX->push_back(i);

            line1a_fd_cast->dataPtX->push_back(i);
            line1b_fd_cast->dataPtX->push_back(i);
            line1c_fd_cast->dataPtX->push_back(i);

            com_dot_x_cast->dataPtX->push_back(i);
            com_dot_y_cast->dataPtX->push_back(i);
            com_dot_z_cast->dataPtX->push_back(i);

            if (frame.unfiltered_joints.size() > 0 && tmp_joints.size() > 0) {
                auto current = frame.unfiltered_joints.begin()->second.at(HAND_RIGHT);
                auto prior = tmp_joints.begin()->second.at(HAND_RIGHT);
                auto duration = frame.durations.at(0);

                auto comdot = frame.com_dots.begin()->second;

                auto vel = frame.filtered_vel.begin()->second.at(HAND_RIGHT);

                line1a_vel_cast->dataPtY->push_back((float)vel.x);
                line1b_vel_cast->dataPtY->push_back((float)vel.y);
                line1c_vel_cast->dataPtY->push_back((float)vel.z);

                line1a_fd_cast->dataPtY->push_back((float)(current.x - prior.x) / duration);
                line1b_fd_cast->dataPtY->push_back((float)(current.y - prior.y) / duration);
                line1c_fd_cast->dataPtY->push_back((float)(current.z - prior.z) / duration);

                com_dot_x_cast->dataPtY->push_back((float)comdot.x);
                com_dot_y_cast->dataPtY->push_back((float)comdot.y);
                com_dot_z_cast->dataPtY->push_back((float)comdot.z);
            } else {
                line1a_vel_cast->dataPtY->push_back(0.0);
                line1b_vel_cast->dataPtY->push_back(0.0);
                line1c_vel_cast->dataPtY->push_back(0.0);

                line1a_fd_cast->dataPtY->push_back(0.0);
                line1b_fd_cast->dataPtY->push_back(0.0);
                line1c_fd_cast->dataPtY->push_back(0.0);

                com_dot_x_cast->dataPtY->push_back(0.0);
                com_dot_y_cast->dataPtY->push_back(0.0);
                com_dot_z_cast->dataPtY->push_back(0.0);
            }

            line1a_vel_cast->updateInternalData();
            line1b_vel_cast->updateInternalData();
            line1c_vel_cast->updateInternalData();

            line1a_fd_cast->updateInternalData();
            line1b_fd_cast->updateInternalData();
            line1c_fd_cast->updateInternalData();

            com_dot_x_cast->updateInternalData();
            com_dot_y_cast->updateInternalData();
            com_dot_z_cast->updateInternalData();

            // Get range of last 1000 points, provided the points in the a axis are in order
            unsigned int minInd = std::max((long)0, (long)line1a_vel_cast->dataPtX->size() - 250);
            unsigned int maxInd = line1a_vel_cast->dataPtX->size() - 1;
            float xmin = (*line1a_vel_cast->dataPtX)[minInd];
            float xmax = (*line1a_vel_cast->dataPtX)[maxInd];
            axesPt->setXAxesLimits(xmin, xmax);
            axesPt2->setXAxesLimits(xmin, xmax);
            i++;
            myplot->Draw();
            window2->postLoopDraw();

            tmp_joints = frame.unfiltered_joints;
        } else {
            std::this_thread::yield();
        }
    }
    /*
    window2->preLoopDraw(true);
    glfwTerminate();
    */
    std::cout << "Finish plotting thread" << std::endl;
}
