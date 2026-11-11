#ifndef LINEAR_TRANSPORT_HPP
#define LINEAR_TRANSPORT_HPP

#include <math.h>
#include "include/Portability.h" // FORCE_INLINE
#include "include/Physics.hpp"   // PhysicsType

// Linear transport: u_t + a u_x = 0
struct LinearTransport {
    static constexpr int NVARS = 1;

    struct Params {
        double a; // Transport velocity
    };

    // f(u) = a*u
    FORCE_INLINE static void flux(const double* U, double* F, const Params& p)
    {
        F[0] = p.a*U[0];
    }

    // |a|
    FORCE_INLINE static double maxSpeed(const double* U, const Params& p)
    {
        (void)U;
        return fabs(p.a);
    }
};
static_assert(PhysicsType<LinearTransport>); // See common/include/Physics.hpp

#endif
