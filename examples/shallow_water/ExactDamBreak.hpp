#ifndef EXACT_DAM_BREAK_HPP
#define EXACT_DAM_BREAK_HPP

#include <math.h>

// Exact dam break for the shallow water equations: water at rest on both sides, hL > hR > 0.
// The dam is at x = 0, and the solution is evaluated at (x, t), t > 0.
//
// From left to right the solution has: the left state, a rarefaction, the star state (hStar, uStar),
// a shock, and the right state.
inline void exactDamBreak(
    double x, double t, double g,
    double hL, double hR,
    double* h, double* u // OUT: Depth and velocity at (x, t)
)
{
    double cL = sqrt(g*hL);

    // Star depth (Newton): the velocity behind the rarefaction must equal the velocity behind the shock
    double hStar = 0.5*(hL + hR);
    for (int it = 0; it < 50; ++it) {
        double s  = sqrt(0.5*g*(1.0/hStar + 1.0/hR));
        double ds = -g/(4.0*hStar*hStar*s);
        double f  = 2.0*(sqrt(g*hStar) - cL) + (hStar - hR)*s;
        double df = sqrt(g/hStar) + s + (hStar - hR)*ds;
        double dh = f/df;
        hStar -= dh;
        if (fabs(dh) < 1e-14*hStar) break;
    }

    double cStar = sqrt(g*hStar);
    double uStar = 2.0*(cL - cStar);
    double shock = hStar*uStar/(hStar - hR); // Shock speed, from conservation of mass

    // Sample the waves at xi = x/t
    double xi = x/t;
    if (xi <= -cL) {               // Left state
        *h = hL;
        *u = 0.0;
    } else if (xi < uStar - cStar) { // Inside the rarefaction
        double c = (2.0*cL - xi)/3.0;
        *h = c*c/g;
        *u = xi + c;
    } else if (xi < shock) {       // Star state
        *h = hStar;
        *u = uStar;
    } else {                       // Right state
        *h = hR;
        *u = 0.0;
    }
}

#endif
