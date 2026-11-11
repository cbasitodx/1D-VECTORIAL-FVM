#ifndef SHALLOW_WATER_HPP
#define SHALLOW_WATER_HPP

#include <math.h>
#include "include/Portability.h" // FORCE_INLINE
#include "include/Physics.hpp"   // PhysicsType

// Shallow water equations with a flat bottom. Requires h > 0.
//
//     U = (h, hu)
//     f(U) = (hu, hu^2/h + g h^2/2)
struct ShallowWater {
    static constexpr int NVARS = 2;

    struct Params {
        double g; // Gravity
    };

    FORCE_INLINE static void flux(const double* U, double* F, const Params& p)
    {
        double u = U[1]/U[0];
        F[0] = U[1];
        F[1] = U[1]*u + 0.5*p.g*U[0]*U[0];
    }

    // |u| + sqrt(g h), with u = hu/h
    FORCE_INLINE static double maxSpeed(const double* U, const Params& p)
    {
        double u = U[1]/U[0];
        return fabs(u) + sqrt(p.g*U[0]);
    }
};
static_assert(PhysicsType<ShallowWater>); // See common/include/Physics.hpp

#endif
