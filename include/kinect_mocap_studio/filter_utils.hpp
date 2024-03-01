#pragma once
#include <Window3dWrapper.h>
#include "BodyTrackingHelpers.h"
#include <k4a/k4atypes.h>
#include <filter/Point.hpp>

template<typename Value>
void add_point(Window3dWrapper& window3d, Point<Value> point, Color color = {1., 0., 0., 1.}, bool stability=false) {
    k4a_float3_t pos;

    pos.v[0] = point.x * 1000;
    pos.v[1] = point.y * 1000;
    pos.v[2] = point.z * 1000;

    k4a_quaternion_t orientation;
    orientation.v[0] = 1;
    orientation.v[1] = 0;
    orientation.v[2] = 0;
    orientation.v[3] = 0;

    window3d.AddJoint(
        pos,
        orientation,
        color,
        stability);
}

template<typename Value>
void add_qtm_point(Window3dWrapper& window3d, Point<Value> point, Color color = {1., 0., 0., 1.}, bool stability=false) {
    k4a_float3_t pos;

    pos.v[0] = point.x * 1000;
    pos.v[1] = point.z * -1000;
    pos.v[2] = point.y * -1000;

    k4a_quaternion_t orientation;
    orientation.v[0] = 1;
    orientation.v[1] = 0;
    orientation.v[2] = 0;
    orientation.v[3] = 0;

    window3d.AddJoint(
        pos,
        orientation,
        color,
        stability);
}

template<typename Value>
void add_theia_point(Window3dWrapper& window3d, Point<Value> point, Color color = {1., 0., 0., 1.}, bool stability=false) {
    k4a_float3_t pos;

    pos.v[0] = point.y * -1000;
    pos.v[1] = point.z * -1000;
    pos.v[2] = point.x * 1000;

    k4a_quaternion_t orientation;
    orientation.v[0] = 1;
    orientation.v[1] = 0;
    orientation.v[2] = 0;
    orientation.v[3] = 0;

    window3d.AddJoint(
        pos,
        orientation,
        color,
        stability);
}

template<typename Value>
void add_bone(Window3dWrapper& window3d, Point<Value> point_a, Point<Value> point_b, Color color = {1., 0., 0., 1.}, bool stability=false) {
    k4a_float3_t pos_a;

    pos_a.v[0] = point_a.x * 1000;
    pos_a.v[1] = point_a.y * 1000;
    pos_a.v[2] = point_a.z * 1000;

    k4a_float3_t pos_b;

    pos_b.v[0] = point_b.x * 1000;
    pos_b.v[1] = point_b.y * 1000;
    pos_b.v[2] = point_b.z * 1000;
    window3d.AddBone(pos_a, pos_b, color, stability);
}

template<typename Value>
void add_qtm_bone(Window3dWrapper& window3d, Point<Value> point_a, Point<Value> point_b, Color color = {1., 0., 0., 1.}, bool stability=false) {
    k4a_float3_t pos_a;

    pos_a.v[0] = point_a.x * 1000;
    pos_a.v[1] = point_a.z * -1000;
    pos_a.v[2] = point_a.y * -1000;

    k4a_float3_t pos_b;

    pos_b.v[0] = point_b.x * 1000;
    pos_b.v[1] = point_b.z * -1000;
    pos_b.v[2] = point_b.y * -1000;
    window3d.AddBone(pos_a, pos_b, color, stability);
}
