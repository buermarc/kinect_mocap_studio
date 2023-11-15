#include <Window3dWrapper.h>
#include "BodyTrackingHelpers.h"
#include <k4a/k4atypes.h>
#include <filter/Point.hpp>

template<typename Value>
void add_point(Window3dWrapper& window3d, Point<Value> point, int color_index) {
    Color color = g_bodyColors[color_index % g_bodyColors.size()];
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
