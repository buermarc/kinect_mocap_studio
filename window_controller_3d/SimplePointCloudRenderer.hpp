// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <mutex>

#include "glad/glad.h"
#include "GLFW/glfw3.h"

#include "linmath.h"
#include "WindowController3dTypes.h"
#include "RendererBase.h"
#include <optional>
#include <vector>

#include <k4a/k4atypes.h>

namespace Visualization
{
    class SimplePointCloudRenderer : public RendererBase
    {
    public:

        SimplePointCloudRenderer();
        ~SimplePointCloudRenderer();
        void Create(GLFWwindow* window)  override;
        void Delete() override;

        void UpdatePointClouds(
            GLFWwindow* window,
            const k4a_float3_t* point3ds,
            uint32_t numPoints,
            uint32_t width, uint32_t height,
            bool useTestPointClouds = false);

        void SetShading(bool enableShading);

        void Render() override;
        void Render(int width, int height);

        void ChangePointCloudSize(float pointCloudSize);

    private:
        // Render settings
        const GLfloat m_defaultPointCloudSize = 0.5f;
        std::optional<GLfloat> m_pointCloudSize;
        bool m_enableShading = false;

        // Point Array Size
        GLsizei m_drawArraySize = 0;

        // Depth Frame Information
        uint32_t m_width = 0;
        uint32_t m_height = 0;

        // OpenGL resources
        GLuint m_vertexArrayObject = 0;
        GLuint m_vertexBufferObject = 0;
        GLfloat m_vertexColorObject[4] = {0.8, 0.8, 0.8, 0.6};

        GLuint m_viewIndex = 0;
        GLuint m_projectionIndex = 0;
        GLuint m_vertexColorIndex = 0;

        // Lock
        std::mutex m_mutex;
    };
}
