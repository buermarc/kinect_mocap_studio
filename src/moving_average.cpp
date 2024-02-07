#include "SampleMathTypes.h"
#include <iostream>
#include <kinect_mocap_studio/moving_average.hpp>

std::optional<Samples::Plane> MovingAverage::get_average(std::optional<Samples::Plane> plane)
{
    std::cout << "Inside here" << std::endl;
    if (plane.has_value()) {
        std::cout << "Before" << std::endl;
        std::cout << plane->Normal.X << std::endl;
        std::cout << plane->Normal.Y << std::endl;
        std::cout << plane->Normal.Z << std::endl;
        std::cout << plane->Origin.X << std::endl;
        std::cout << plane->Origin.Y << std::endl;
        std::cout << plane->Origin.Z << std::endl;

        m_fill = std::min(++m_fill, m_window_size);

        if (!m_init) {
            m_init = true;
            Samples::Plane& current = m_window[m_idx];
            current.Normal.X = plane->Normal.X;
            current.Normal.Y = plane->Normal.Y;
            current.Normal.Z = plane->Normal.Z;
            current.Origin.X = plane->Origin.X;
            current.Origin.Y = plane->Origin.Y;
            current.Origin.Z = plane->Origin.Z;
            current.C = plane->C;
            m_idx = (m_idx + 1) % m_window_size;
            m_latest = plane;
            std::cout << "Fin" << std::endl;
            return plane;
        }

        Samples::Plane& current = m_window[m_idx];

        std::cout << "Current" << std::endl;
        std::cout << current.Normal.X << std::endl;
        std::cout << current.Normal.Y << std::endl;
        std::cout << current.Normal.Z << std::endl;
        std::cout << current.Origin.X << std::endl;
        std::cout << current.Origin.Y << std::endl;
        std::cout << current.Origin.Z << std::endl;
        std::cout << current.C << std::endl;

        current.Normal.X = plane->Normal.X;
        current.Normal.Y = plane->Normal.Y;
        current.Normal.Z = plane->Normal.Z;
        current.Origin.X = plane->Origin.X;
        current.Origin.Y = plane->Origin.Y;
        current.Origin.Z = plane->Origin.Z;
        current.C = plane->C;

        for (int i = 1; i < m_fill; ++i) {
            int offset = (i) % m_window_size;
            Samples::Plane& current = m_window[offset];
            plane->Normal.X += current.Normal.X;
            plane->Normal.Y += current.Normal.Y;
            plane->Normal.Z += current.Normal.Z;
            plane->Origin.X += current.Origin.X;
            plane->Origin.Y += current.Origin.Y;
            plane->Origin.Z += current.Origin.Z;
            plane->C += current.C;
        }
        plane->Normal.X /= m_fill;
        plane->Normal.Y /= m_fill;
        plane->Normal.Z /= m_fill;
        plane->Origin.X /= m_fill;
        plane->Origin.Y /= m_fill;
        plane->Origin.Z /= m_fill;
        plane->C /= m_fill;

        m_idx = (m_idx + 1) % m_window_size;
        m_latest = plane;
        std::cout << "Fin" << std::endl;
        std::cout << "After" << std::endl;
        std::cout << plane->Normal.X << std::endl;
        std::cout << plane->Normal.Y << std::endl;
        std::cout << plane->Normal.Z << std::endl;
        std::cout << plane->Origin.X << std::endl;
        std::cout << plane->Origin.Y << std::endl;
        std::cout << plane->Origin.Z << std::endl;
        return plane;
    }
    std::cout << "Fin" << std::endl;
    return m_latest;
}
