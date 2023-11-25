// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "SimplePointCloudRenderer.hpp"

#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>
#include <algorithm>
#include <array>
#include <thread>

#include "SimplePointCloudShaders.hpp"
#include "ViewControl.h"
#include "Helpers.h"

using namespace linmath;
using namespace Visualization;

PointCloudVertex testVertices[] =
{
    {{-0.5f, -0.5f, -2.5f}, {1.0f, 0.0f, 0.0f, 1.0f}, {10, 0}},
    {{ 0.5f, -0.5f, -2.5f}, {0.0f, 1.0f, 0.0f, 1.0f}, {20, 0}},
    {{-0.5f,  0.5f, -2.5f}, {0.0f, 0.0f, 1.0f, 1.0f}, {30, 0}},
    {{ 0.5f,  0.5f, -2.5f}, {1.0f, 1.0f, 0.0f, 1.0f}, {40, 0}},

    {{-0.5f, -0.5f, -3.5f}, {0.0f, 1.0f, 1.0f, 1.0f}, {50, 0}},
    {{ 0.5f, -0.5f, -3.5f}, {1.0f, 0.0f, 1.0f, 1.0f}, {60, 0}},
    {{-0.5f,  0.5f, -3.5f}, {1.0f, 1.0f, 0.5f, 1.0f}, {70, 0}},
    {{ 0.5f,  0.5f, -3.5f}, {0.5f, 0.5f, 1.0f, 1.0f}, {80, 0}}
};

SimplePointCloudRenderer::SimplePointCloudRenderer()
{
    mat4x4_identity(m_view);
    mat4x4_identity(m_projection);
}

SimplePointCloudRenderer::~SimplePointCloudRenderer()
{
    Delete();
}

void SimplePointCloudRenderer::Create(GLFWwindow* window)
{
    CheckAssert(!m_initialized);
    m_initialized = true;

    m_window = window;
    glfwMakeContextCurrent(window);

    // Context Settings
    glEnable(GL_PROGRAM_POINT_SIZE);

    m_vertexShader = glCreateShader(GL_VERTEX_SHADER);
    const GLchar* vertexShaderSources[] = { glslShaderVersion, glslPointCloudVertexShader };
    int numVertexShaderSources = sizeof(vertexShaderSources) / sizeof(*vertexShaderSources);
    glShaderSource(m_vertexShader, numVertexShaderSources, vertexShaderSources, NULL);
    glCompileShader(m_vertexShader);
    ValidateShader(m_vertexShader);

    m_fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    const GLchar* fragmentShaderSources[] = { glslShaderVersion, glslPointCloudFragmentShader};
    int numFragmentShaderSources = sizeof(fragmentShaderSources) / sizeof(*fragmentShaderSources);
    glShaderSource(m_fragmentShader, numFragmentShaderSources, fragmentShaderSources, NULL);
    glCompileShader(m_fragmentShader);
    ValidateShader(m_fragmentShader);

    m_shaderProgram = glCreateProgram();
    glAttachShader(m_shaderProgram, m_vertexShader);
    glAttachShader(m_shaderProgram, m_fragmentShader);
    glLinkProgram(m_shaderProgram);
    ValidateProgram(m_shaderProgram);

    glGenVertexArrays(1, &m_vertexArrayObject);
    glBindVertexArray(m_vertexArrayObject);
    glGenBuffers(1, &m_vertexBufferObject);
    m_viewIndex = glGetUniformLocation(m_shaderProgram, "view");
    m_projectionIndex = glGetUniformLocation(m_shaderProgram, "projection");
    m_vertexColorIndex = glGetUniformLocation(m_shaderProgram, "vertexColor");
}

void SimplePointCloudRenderer::Delete()
{
    if (!m_initialized)
    {
        return;
    }

    m_initialized = false;
    glDeleteBuffers(1, &m_vertexBufferObject);

    glDeleteShader(m_vertexShader);
    glDeleteShader(m_fragmentShader);
    glDeleteProgram(m_shaderProgram);
}

void SimplePointCloudRenderer::UpdatePointClouds(
    GLFWwindow* window,
    const k4a_float3_t* point3ds,
    uint32_t numPoints,
    uint32_t width, uint32_t height,
    bool useTestPointClouds)
{
    if (window != m_window)
    {
        Create(window);
    }

    if (m_width != width && m_height != height)
    {
        Fail("Width and Height (%u, %u) does not match the DepthXYTable settings: (%u, %u) are expected!", width, height, m_width, m_height);
    }

    glBindVertexArray(m_vertexArrayObject);
    // Create buffers and bind the geometry
    glBindBuffer(GL_ARRAY_BUFFER, m_vertexBufferObject);

    if (!useTestPointClouds)
    {
        glBufferData(GL_ARRAY_BUFFER, numPoints * sizeof(k4a_float3_t), point3ds, GL_STREAM_DRAW);
    }
    else
    {
        glBufferData(GL_ARRAY_BUFFER, sizeof(testVertices), testVertices, GL_STREAM_DRAW);
    }

    // Set the vertex attribute pointers
    // Vertex Positions
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(k4a_float3_t), (void*)0);

    glBindVertexArray(0);

    m_drawArraySize = useTestPointClouds ? 8 : GLsizei(numPoints);
}

void SimplePointCloudRenderer::Render()
{
    std::array<int, 4> data; // x, y, width, height

    glGetIntegerv(GL_VIEWPORT, data.data());
    Render(data[2], data[3]);
}

void SimplePointCloudRenderer::Render(int width, int height)
{
    glEnable(GL_DEPTH_TEST);
    // Enable blending
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    float pointSize;
    if (m_pointCloudSize)
    {
        pointSize = m_pointCloudSize.value();
    }
    else if (m_width == 0 || m_height == 0)
    {
        pointSize = m_defaultPointCloudSize;
    }
    else
    {
        pointSize = std::min(2.f * width / (float)m_width, 2.f * height / (float)m_height);
    }
    glPointSize(pointSize);

    glUseProgram(m_shaderProgram);

    // Update model/view/projective matrices in shader
    glUniformMatrix4fv(m_viewIndex, 1, GL_FALSE, (const GLfloat*)m_view);
    glUniformMatrix4fv(m_projectionIndex, 1, GL_FALSE, (const GLfloat*)m_projection);

    // Update render settings in shader
    glUniform4fv(m_vertexColorIndex, 1, (GLfloat*)m_vertexColorObject);

    // Render point cloud
    glBindVertexArray(m_vertexArrayObject);
    glDrawArrays(GL_POINTS, 0, m_drawArraySize);
    glBindVertexArray(0);
}

void SimplePointCloudRenderer::ChangePointCloudSize(float pointCloudSize)
{
    std::lock_guard<std::mutex> lock(m_mutex);

    m_pointCloudSize = pointCloudSize;
}
