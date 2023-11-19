#pragma once
#include <Window3dWrapper.h>
#include <k4a/k4atypes.h>
#include <k4abttypes.h>

#include <filter/SkeletonFilter.hpp>

void visualizeResult(k4abt_frame_t bodyFrame, Window3dWrapper& window3d, int depthWidth, int depthHeight, SkeletonFilterBuilder<double> builder, uint64_t timestamp);
