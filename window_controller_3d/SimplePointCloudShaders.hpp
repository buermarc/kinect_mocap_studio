// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "GlShaderDefs.h"

// ************** Point Cloud Vertex Shader **************
static const char* const glslPointCloudVertexShader = GLSL_STRING(

    layout(location = 0) in vec3 vertexPosition;

    out vec4 fragmentColor;

    uniform mat4 view;
    uniform mat4 projection;
    uniform vec4 vertexColor;

    void main()
    {
        gl_Position = projection * view * vec4(vertexPosition, 1);

        fragmentColor = vertexColor;
    }

);  // GLSL_STRING


// ************** Point Cloud Fragment Shader **************
static const char* const glslPointCloudFragmentShader = GLSL_STRING(

    out vec4 fragColor;
    in vec4 fragmentColor;

    void main()
    {
        fragColor = fragmentColor;
    }

);  // GLSL_STRING
