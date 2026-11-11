#ifndef EULER_HPP
#define EULER_HPP

#include <math.h>
#include "include/Portability.h" // FORCE_INLINE
#include "include/Physics.hpp"   // PhysicsType

// 1D Euler equations for an ideal gas. Requires rho > 0 and p > 0.
//
//     U = (rho, rho u, E)
//     p = (gamma - 1)*(E - rho u^2/2)
//     f(U) = (rho u, rho u^2 + p, u*(E + p))
struct Euler {
    static constexpr int NVARS = 3;

    struct Params {
        double gamma; // Ratio of specific heats
    };

    FORCE_INLINE static void flux(const double* U, double* F, const Params& p)
    {
        double invRho   = 1.0/U[0]; // Computed once: divisions are expensive
        double u        = U[1]*invRho;
        double pressure = (p.gamma - 1.0)*(U[2] - 0.5*U[1]*u);
        F[0] = U[1];
        F[1] = U[1]*u + pressure;
        F[2] = u*(U[2] + pressure);
    }

    // |u| + c, with c = sqrt(gamma p/rho)
    FORCE_INLINE static double maxSpeed(const double* U, const Params& p)
    {
        double invRho   = 1.0/U[0];
        double u        = U[1]*invRho;
        double pressure = (p.gamma - 1.0)*(U[2] - 0.5*U[1]*u);
        return fabs(u) + sqrt(p.gamma*pressure*invRho);
    }
};
static_assert(PhysicsType<Euler>); // See common/include/Physics.hpp

#endif
