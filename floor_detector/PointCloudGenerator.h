// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <k4a/k4atypes.h>
#include "WindowController3dTypes.h"

#include <vector>
#include <tuple>

namespace Samples
{
    class PointCloudGenerator
    {
    public:
        PointCloudGenerator(const k4a_calibration_t& sensorCalibration);
        ~PointCloudGenerator();

        void Update(k4a_image_t depthImage);
        const std::vector<k4a_float3_t>& GetCloudPoints(int downsampleStep = 1);
        std::tuple<std::vector<Visualization::PointCloudVertex>, std::vector<uint16_t>> GetRenderCapableCloudPoints(k4a_image_t depthImage);
        std::vector<uint16_t> UpdateDepthBuffer(k4a_image_t depthFrame);

    private:
        k4a_transformation_t m_transformationHandle = nullptr;
        k4a_image_t m_pointCloudImage_int16x3 = nullptr;
        std::vector<k4a_float3_t> m_cloudPoints;
        std::vector<uint16_t> m_depthBuffer;
        std::vector<Visualization::PointCloudVertex> m_pointClouds;
    };
}
