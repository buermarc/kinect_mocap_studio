#pragma once
#include <Window3dWrapper.h>
#include "BodyTrackingHelpers.h"
#include <k4a/k4atypes.h>
#include <filter/Point.hpp>

template<typename Value>
void add_point(Window3dWrapper& window3d, Point<Value> point, Color color = {1., 0., 0., 1.}) {
    k4a_float3_t pos;

    pos.v[0] = point.x;
    pos.v[1] = point.y;
    pos.v[2] = point.z;

    k4a_quaternion_t orientation;
    orientation.v[0] = 1;
    orientation.v[1] = 0;
    orientation.v[2] = 0;
    orientation.v[3] = 0;

    window3d.AddJoint(
        pos,
        orientation,
        color);
}

template<typename Value>
void add_bone(Window3dWrapper& window3d, Point<Value> point_a, Point<Value> point_b, Color color = {1., 0., 0., 1.}) {
    k4a_float3_t pos_a;

    pos_a.v[0] = point_a.x;
    pos_a.v[1] = point_a.y;
    pos_a.v[2] = point_a.z;

    k4a_float3_t pos_b;

    pos_b.v[0] = point_b.x;
    pos_b.v[1] = point_b.y;
    pos_b.v[2] = point_b.z;
    window3d.AddBone(pos_a, pos_b, color);
}
