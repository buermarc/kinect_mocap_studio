#include <kinect_mocap_studio/filter_utils.hpp>
#include <kinect_mocap_studio/plotting.hpp>
#include <kinect_mocap_studio/queues.hpp>
#include <kinect_mocap_studio/utils.hpp>
#include <kinect_mocap_studio/cli.hpp>

#include <optional>
#include <thread>
#include <iostream>
#include <future>
#include <memory>

#include <k4abt.h>
#include <k4a/k4a.h>
#include <k4a/k4atypes.h>
#include <k4abttypes.h>

#include "BodyTrackingHelpers.h"

#include "FloorDetector.h"
#include "GLPL/rendering/IDrawable.h"
#include "GLPL/rendering/ShaderSet.h"
#include "linmath.h"
#include <Window3dWrapper.h>

#include <filter/com.hpp>
#include <GLPL/plot/plot.h>
#include <GLPL/window/window.h>
#include <glm/gtx/string_cast.hpp>


void plotThread() {
    while (!s_glfwInitialized) {
        std::this_thread::yield();
    }
    ProcessedFrame frame;

    int windowWidth  = 1600;
    int windowHeight = 800;
    std::shared_ptr<GLPL::IWindow> window = std::shared_ptr<GLPL::IWindow>(new GLPL::Window(windowWidth, windowHeight,  false, false));
    std::shared_ptr<GLPL::Window> window2 = std::dynamic_pointer_cast<GLPL::Window>(window);
    std::vector<float> xVec12 = {};
    std::vector<float> yVec12 = {};
    xVec12.reserve(2000);
    yVec12.reserve(2000);

    // glm::mat4 parentTransform(1.0);
    // std::cout << "Parent transform." << std::endl;
    // std::cout << glm::to_string(parentTransform) << std::endl;
    std::shared_ptr<GLPL::Plot> myplot = std::make_shared<GLPL::Plot>(0.0, 0.0, 1.0, 1.0, window2->getParentDimensions(), 2, 1);
    // std::shared_ptr<GLPL::Plot> myplot = std::make_shared<GLPL::Plot>(0.0, 0.0, 1.0, 1.0, std::make_shared<GLPL::ParentDimensions>(GLPL::ParentDimensions{
      //   parentTransform, 1, 1, 1024, 1024, std::make_shared<GLPL::ShaderSet>()
    // }), 2, 1);
    std::shared_ptr<GLPL::IDrawable> myPlotPt = std::dynamic_pointer_cast<GLPL::IDrawable>(myplot);
    window2->addPlot(myPlotPt);

    std::shared_ptr<GLPL::Axes2D> axesPt = std::dynamic_pointer_cast<GLPL::Axes2D>(myplot->getAxes(0));
    std::shared_ptr<GLPL::ILine2D> line12 = axesPt->addLine(&xVec12, &yVec12, GLPL::SINGLE_LINE, LC_YELLOW, 0.5, "Underdamped 1");
    std::shared_ptr<GLPL::Line2D2Vecs> line12b = std::dynamic_pointer_cast<GLPL::Line2D2Vecs>(line12);

    axesPt->setAxesBoxOn(false);
    axesPt->setButtonState("Grid", false);
    axesPt->setXLabel("Time (s)");
    axesPt->setYLabel("Displacement (m)");
    axesPt->setTitle("Spring Damping Over Time");
    axesPt->setYLabelRotation(GLPL::SIDEWAYS_RIGHT);
    axesPt->setButtonState("X Axes Limits Scaling", false);
    axesPt->setButtonState("Y Axes Limits Scaling", true);
    axesPt->showLegend(true);

    float yVal12 = 0;
    float i =0;

    bool skip = false;
    while (s_isRunning) {
        // bool retrieved =  processed_queue.Consume(frame);
        bool retrieved = true;
        if (retrieved) {
            if (skip) {
                std::cout << "Viz is skipping" << std::endl;
                skip = false;
                continue;
            }
            window2->preLoopDraw(true);
            line12b->dataPtX->push_back(i);
            yVal12 = cos((i) / (25*M_PI)) * exp(-(i)/(25*8*M_PI));
            line12b->dataPtY->push_back(yVal12);
            std::cout << "yval12 " << yVal12 << std::endl;
            line12b->updateInternalData();
            //
            // Get range of last 1000 points, provided the points in the a axis are in order
            unsigned int minInd = std::max((long)0, (long)line12b->dataPtX->size() - 1000);
            unsigned int maxInd = line12b->dataPtX->size()-1;
            float xmin = (*line12b->dataPtX)[minInd];
            float xmax = (*line12b->dataPtX)[maxInd];
            axesPt->setXAxesLimits(xmin, xmax);
            i++;
            myplot->Draw();
            window2->postLoopDraw();

        } else {
            std::this_thread::yield();
        }
    }
    std::cout << "Finish plot thread" << std::endl;
}
