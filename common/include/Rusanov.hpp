#ifndef RUSANOV_HPP
#define RUSANOV_HPP

#include "include/Portability.h" // FORCE_INLINE

// The Rusanov numerical flux. It knows nothing about the equation: it only receives values that the
// solver has already computed with the physics. For variable k, at the interface between
// the cells L and R:
//
//     F_k = (f_k(uR) + f_k(uL))/2 - max(sL, sR)/2 * (uR_k - uL_k) ; k = 1, ..., M
//
// where sL and sR are the maximum wave speeds of each cell. It is split in two functions so that
// the speed term is computed once per interface and then used for the M variables.
struct Rusanov {
    // max(sL, sR)/2
    // The ?: only chooses between two values (no computation inside), so the compiler can do it without a branch
    // and vectorize the solver loop. Multiplying by 0.5 gives exactly the same result as dividing by 2
    FORCE_INLINE static double halfSpeed(double sL, double sR)
    {
        return 0.5*(sL > sR ? sL : sR);
    }

    // The numerical flux of one variable
    FORCE_INLINE static double F(
        double uL,       // IN: Variable at the left cell
        double uR,       // IN: Variable at the right cell
        double fL,       // IN: Flux of the variable at the left cell
        double fR,       // IN: Flux of the variable at the right cell
        double halfSpeed // IN: Result of Rusanov::halfSpeed
    )
    {
        return (fR + fL)/2 - halfSpeed * (uR - uL);
    }
};

#endif
