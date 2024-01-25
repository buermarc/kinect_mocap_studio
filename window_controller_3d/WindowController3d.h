// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <array>
#include <mutex>

#include "glad/glad.h"
#include "GLFW/glfw3.h"

#include "linmath.h"

#include "ViewControl.h"
#include "SimplePointCloudRenderer.hpp"
#include "SkeletonRenderer.h"
#include "FloorRenderer.h"
#include "BosRenderer.h"

#include <k4a/k4atypes.h>

namespace Visualization
{
    typedef int64_t(*CloseCallbackType)(void* context);
    // https://www.glfw.org/docs/latest/group__keys.html
    typedef int64_t(*KeyCallbackType)(void* context, int glfwKey);

    enum class SkeletonRenderMode
    {
        DefaultRender = 0,
        SkeletonOverlay,
        SkeletonOverlayWithJointFrame
    };

    enum class Layout3d
    {
        OnlyMainView = 0,
        FourViews,
        MainAndBosView,
        Count,
    };

    class WindowController3d
    {
    public:
        WindowController3d();

        void Create(
            const char *name,
            bool showWindow = true,
            int width = -1,
            int height = -1,
            bool fullscreen = false);

        void Delete();

        void SetWindowPosition(int xPos, int yPos);

        // Initialize the point cloud renderer
        // If you want to enable the point cloud shading for better visualization, you need to pass in the DepthXY table
        bool InitializePointCloudRenderer(
            bool enableShading,
            const float* depthXyTableInterleaved,
            int width, int height);

        void UpdatePointClouds(
            const k4a_float3_t* point3d,
            uint32_t numPoints,
            uint32_t width, uint32_t height,
            bool useTestPointClouds = false);

        void CleanJointsAndBones();

        void AddJoint(const Visualization::Joint& joint, bool stability=false);

        void AddBone(const Visualization::Bone& bone, bool stability=false);

        void PreRender();

        void Render(
            std::vector<uint8_t>* renderedPixelsBgr = nullptr,
            int* pixelsWidth = nullptr,
            int* pixelsHeight = nullptr);

        void SetPointCloudShading(bool enableShading);

        void SetDefaultVerticalFOV(float degrees);

        void SetMirrorMode(bool enableMirrorMode);

        void SetSkeletonRenderMode(SkeletonRenderMode skeletonRenderMode);

        void SetLayout3d(Layout3d layout3d);

        void ChangePointCloudSize(float pointCloudSize);

        void SetFloorRendering(bool enableFloorRendering, linmath::vec3 floorPosition, linmath::quaternion floorOrientation);
        void OldSetBosRendering(bool enableBosRendering, linmath::vec3 bosPosition, linmath::quaternion bosOrientation);
        void SetBosRendering(bool enableBosRendering, linmath::vec3 a, linmath::vec3 b, linmath::vec3 c, linmath::vec3 d);
        void DisableBosRendering();

        // Methods to set external callback functions
        void SetCloseCallback(CloseCallbackType callback, void* context);

        void SetKeyCallback(KeyCallbackType callback, void* context);
        void Scroll(float offset);
        void SetTopViewPoint();

    protected:
        void FrameBufferSizeCallback(GLFWwindow* window, int width, int height);
        void MouseButtonCallback(GLFWwindow* window, int button, int action, int mods);
        void MouseMovementCallback(GLFWwindow* window, double xpos, double ypos);
        void MouseScrollCallback(GLFWwindow* window, double xoffset, double yoffset);
        void KeyPressCallback(GLFWwindow* window, int key, int scancode, int action, int mods);
        void WindowCloseCallback(GLFWwindow* window);

    private:
        void RenderScene(ViewControl& viewControl, Viewport viewport);
        void TriggerCameraPivotPointRendering();
        void ChangeCameraPivotPoint(ViewControl& viewControl, linmath::vec2 screenPos);
        void GetCursorPosInScreenCoordinates(GLFWwindow* window, linmath::vec2 outScreenPos);
        void GetCursorPosInScreenCoordinates(double cursorPosX, double cursorPosY, linmath::vec2 outScreenPos);
        void UpdateRenderersViewProjection(linmath::mat4x4 view, linmath::mat4x4 projection);

        bool m_initialized = false;

        // WindowController3d Settings
        Layout3d m_layout3d = Layout3d::OnlyMainView;
        SkeletonRenderMode m_skeletonRenderMode = SkeletonRenderMode::DefaultRender;
        bool m_enableFloorRendering = false;
        bool m_enableBosRendering = false;
        bool m_onlyStability = false;

        // View Controls
        ViewControl m_viewControl;
        ViewControl m_leftViewControl;
        ViewControl m_rightViewControl;
        ViewControl m_topViewControl;
        ViewControl m_realTopViewControl;
        std::array<ViewControl*,5> m_allViewControls = { &m_viewControl, &m_leftViewControl, &m_rightViewControl, &m_topViewControl, &m_realTopViewControl };

        // Object renderers
        SimplePointCloudRenderer m_pointCloudRenderer;
        SkeletonRenderer m_skeletonRenderer;
        SkeletonRenderer m_stabilityMetricsRenderer;
        FloorRenderer m_floorRenderer;
        FloorRenderer m_old_bosRenderer;
        BosRenderer m_bosRenderer;

        // Render time information
        double m_lastFrame = 0.;
        float m_deltaTime = 0.f;

        // Window information
        int m_windowWidth;
        int m_windowHeight;
        int m_windowStartPositionX;  // the upper-left corner of the window position
        int m_windowStartPositionY;  // the upper-left corner of the window position
        float m_defaultWindowWidthRatio = 0.6f;  // Default window width ratio relative to the full display screen
        float m_defaultWindowHeightRatio = 0.6f;  // Default window height ratio relative to the full display screen

        // OpenGL resources
        GLFWwindow* m_window = nullptr;

        // Input status
        bool m_mouseButtonLeftPressed = false;
        bool m_mouseButtonRightPressed = false;
        int m_cameraPivotPointRenderCount = 0;
        linmath::vec2 m_prevMouseScreenPos = { 0.f, 0.f };

        // External Callback functions
        CloseCallbackType m_closeCallback = nullptr;
        void *m_closeCallbackContext = nullptr;
        KeyCallbackType m_keyCallback = nullptr;
        void *m_keyCallbackContext = nullptr;

        // Lock
        std::mutex m_mutex;
    };
}
