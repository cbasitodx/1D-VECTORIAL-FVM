#ifndef BURGERS_HPP
#define BURGERS_HPP

#include <math.h>
#include "include/Portability.h" // FORCE_INLINE
#include "include/Physics.hpp"   // PhysicsType

// Burgers equation: u_t + (u^2/2)_x = 0
struct Burgers {
    static constexpr int NVARS = 1;

    struct Params {}; // No parameters

    // f(u) = u^2/2
    FORCE_INLINE static void flux(const double* U, double* F, const Params& p)
    {
        (void)p;
        F[0] = 0.5*U[0]*U[0];
    }

    // |u|
    FORCE_INLINE static double maxSpeed(const double* U, const Params& p)
    {
        (void)p;
        return fabs(U[0]);
    }
};
static_assert(PhysicsType<Burgers>); // See common/include/Physics.hpp

#endif
