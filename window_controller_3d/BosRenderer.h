// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <vector>

#include "glad/glad.h"
#include "GLFW/glfw3.h"

#include "RendererBase.h"
#include "WindowController3dTypes.h"
#include "linmath.h"

namespace Visualization
{
    class BosRenderer : public RendererBase
    {
    public:
        BosRenderer();
        void SetBosPlacement(linmath::vec3 a, linmath::vec3 b, linmath::vec3 c, linmath::vec3 d);

        // Renderer functions
        void Create(GLFWwindow* window) override;
        void Delete() override;

        void Render() override;

        void setColor(linmath::vec4 color);

    private:
        void BuildVertices();

        void UpdateVAO();

        void AddIndices(uint32_t i1, uint32_t i2, uint32_t i3);

        linmath::mat4x4 m_model;

        // Settings
        linmath::vec4 m_color;

        linmath::vec3 m_a, m_b, m_c, m_d;

        // Data buffers
        std::vector<MonoVertex> m_vertices;
        std::vector<uint32_t> m_indices;

        // OpenGL objects
        GLuint m_vertexArrayObject;
        GLuint m_vertexBufferObject;
        GLuint m_elementBufferObject;

        GLuint m_modelIndex;
        GLuint m_viewIndex;
        GLuint m_projectionIndex;

        GLuint m_colorIndex;
    };
}
