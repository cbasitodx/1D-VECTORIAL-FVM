#ifndef MHD_HPP
#define MHD_HPP

#include <math.h>
#include "include/Portability.h" // FORCE_INLINE
#include "include/Physics.hpp"   // PhysicsType

// 1D ideal MHD equations (Castro-Díaz et al., "A second order PVM flux limiter method", 2014, eq. (4.1)).
// Units with mu_0 = 1: the magnetic pressure is B^2/2. Requires rho > 0 and P > 0.
//
//     U    = (rho, rho vx, rho vy, rho vz, Bx, By, Bz, E)
//     f(U) = (rho vx,
//             rho vx^2 + P* - Bx^2,
//             rho vx vy - Bx By,
//             rho vx vz - Bx Bz,
//             0,
//             vx By - vy Bx,
//             vx Bz - vz Bx,
//             vx (E + P*) - Bx (vx Bx + vy By + vz Bz))
//
// with  E  = rho q^2/2 + B^2/2 + P/(gamma - 1)   (q^2 = vx^2 + vy^2 + vz^2, B^2 = Bx^2 + By^2 + Bz^2)
// and   P* = P + B^2/2                           (total pressure: hydrostatic + magnetic)
//
// Bx is kept as a variable, as in the paper, although its flux is 0: it never changes.
struct MHD {
    static constexpr int NVARS = 8;

    struct Params {
        double gamma; // Adiabatic constant (ratio of specific heats)
    };

    FORCE_INLINE static void flux(const double* U, double* F, const Params& p)
    {
        double invRho = 1.0/U[0]; // Computed once: divisions are expensive
        double vx = U[1]*invRho, vy = U[2]*invRho, vz = U[3]*invRho;
        double Bx = U[4],        By = U[5],        Bz = U[6];
        double E  = U[7];

        double B2       = Bx*Bx + By*By + Bz*Bz;
        double rhoQ2    = U[1]*vx + U[2]*vy + U[3]*vz;               // rho q^2
        double pressure = (p.gamma - 1.0)*(E - 0.5*rhoQ2 - 0.5*B2);  // Hydrostatic pressure P
        double pStar    = pressure + 0.5*B2;                         // Total pressure P*
        double vB       = vx*Bx + vy*By + vz*Bz;

        F[0] = U[1];
        F[1] = U[1]*vx + pStar - Bx*Bx;
        F[2] = U[1]*vy - Bx*By;
        F[3] = U[1]*vz - Bx*Bz;
        F[4] = 0.0;
        F[5] = vx*By - vy*Bx;
        F[6] = vx*Bz - vz*Bx;
        F[7] = vx*(E + pStar) - Bx*vB;
    }

    // |vx| + c_f, with the fast magnetosonic speed
    //     c_f^2 = (a^2 + b^2 + sqrt((a^2 + b^2)^2 - 4 a^2 bx^2))/2,   a^2 = gamma P/rho,  b = B/sqrt(rho)
    FORCE_INLINE static double maxSpeed(const double* U, const Params& p)
    {
        double invRho = 1.0/U[0];
        double vx = U[1]*invRho, vy = U[2]*invRho, vz = U[3]*invRho;
        double Bx = U[4],        By = U[5],        Bz = U[6];

        double B2       = Bx*Bx + By*By + Bz*Bz;
        double rhoQ2    = U[1]*vx + U[2]*vy + U[3]*vz;
        double pressure = (p.gamma - 1.0)*(U[7] - 0.5*rhoQ2 - 0.5*B2);

        double a2  = p.gamma*pressure*invRho; // Sound speed squared
        double b2  = B2*invRho;               // Alfven speed squared (whole field)
        double bx2 = Bx*Bx*invRho;            // Alfven speed squared (along x)
        double sum = a2 + b2;
        // The discriminant is (a^2 - b^2)^2 + 4 a^2 (b^2 - bx^2) >= 0, but rounding can make it slightly negative
        // when it is ~0. The ?: only chooses between two values (no computation inside), so it does not stop vectorization
        double disc = sum*sum - 4.0*a2*bx2;
        disc = disc > 0.0 ? disc : 0.0;
        double cf = sqrt(0.5*(sum + sqrt(disc)));
        return fabs(vx) + cf;
    }

    // Initial conditions are given in primitive variables W = (rho, vx, vy, vz, Bx, By, Bz, P).
    // Not used by the solver (not part of PhysicsType): helpers for the tests
    static void primitiveToConserved(const double* W, double* U, const Params& p)
    {
        double rho = W[0], vx = W[1], vy = W[2], vz = W[3], Bx = W[4], By = W[5], Bz = W[6], P = W[7];
        U[0] = rho;
        U[1] = rho*vx;
        U[2] = rho*vy;
        U[3] = rho*vz;
        U[4] = Bx;
        U[5] = By;
        U[6] = Bz;
        U[7] = 0.5*rho*(vx*vx + vy*vy + vz*vz) + 0.5*(Bx*Bx + By*By + Bz*Bz) + P/(p.gamma - 1.0);
    }

    static void conservedToPrimitive(const double* U, double* W, const Params& p)
    {
        double rho = U[0];
        double vx = U[1]/rho, vy = U[2]/rho, vz = U[3]/rho;
        double Bx = U[4],     By = U[5],     Bz = U[6];
        W[0] = rho; W[1] = vx; W[2] = vy; W[3] = vz; W[4] = Bx; W[5] = By; W[6] = Bz;
        W[7] = (p.gamma - 1.0)*(U[7] - 0.5*rho*(vx*vx + vy*vy + vz*vz) - 0.5*(Bx*Bx + By*By + Bz*Bz));
    }
};
static_assert(PhysicsType<MHD>); // See common/include/Physics.hpp

#endif
