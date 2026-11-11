#ifndef EXACT_RIEMANN_HPP
#define EXACT_RIEMANN_HPP

#include <math.h>

// Pressure function of one side of the Riemann problem, and its derivative. Gives the velocity jump
// across the wave of that side when the star pressure is p: a shock if p > pK, a rarefaction otherwise.
inline void riemannPressureFunction(double p, double gamma, double rhoK, double pK, double cK, double* f, double* df)
{
    if (p > pK) {
        double A = 2.0/((gamma + 1.0)*rhoK);
        double B = (gamma - 1.0)/(gamma + 1.0)*pK;
        double s = sqrt(A/(p + B));
        *f  = (p - pK)*s;
        *df = s*(1.0 - 0.5*(p - pK)/(p + B));
    } else {
        *f  = 2.0*cK/(gamma - 1.0)*(pow(p/pK, (gamma - 1.0)/(2.0*gamma)) - 1.0);
        *df = pow(p/pK, -(gamma + 1.0)/(2.0*gamma))/(rhoK*cK);
    }
}

// The solution on the left side of the contact discontinuity, for the state (rhoK, uK, pK) of that side.
// The right side uses the same function by mirroring it: x -> -x and u -> -u.
inline void riemannSampleLeftSide(
    double xi, double gamma,
    double rhoK, double uK, double pK, double cK,
    double pStar, double uStar,
    double* rho, double* u, double* p // OUT: Density, velocity and pressure at xi = x/t
)
{
    if (pStar > pK) { // Shock
        double ratio = pStar/pK;
        double m     = (gamma - 1.0)/(gamma + 1.0);
        double shock = uK - cK*sqrt((gamma + 1.0)/(2.0*gamma)*ratio + (gamma - 1.0)/(2.0*gamma));
        if (xi < shock) {
            *rho = rhoK; *u = uK; *p = pK;
        } else {
            *rho = rhoK*(ratio + m)/(m*ratio + 1.0); *u = uStar; *p = pStar;
        }
    } else {          // Rarefaction
        double cStar = cK*pow(pStar/pK, (gamma - 1.0)/(2.0*gamma));
        if (xi < uK - cK) {            // Before the rarefaction
            *rho = rhoK; *u = uK; *p = pK;
        } else if (xi > uStar - cStar) { // After the rarefaction
            *rho = rhoK*pow(pStar/pK, 1.0/gamma); *u = uStar; *p = pStar;
        } else {                       // Inside the rarefaction
            double c = 2.0/(gamma + 1.0)*(cK + 0.5*(gamma - 1.0)*(uK - xi));
            *u   = 2.0/(gamma + 1.0)*(cK + 0.5*(gamma - 1.0)*uK + xi);
            *rho = rhoK*pow(c/cK, 2.0/(gamma - 1.0));
            *p   = pK*pow(c/cK, 2.0*gamma/(gamma - 1.0));
        }
    }
}

// Exact solution of the Riemann problem for the Euler equations, with the left state (rhoL, uL, pL)
// and the right state (rhoR, uR, pR). The discontinuity is at x = 0, and the solution is evaluated at (x, t), t > 0.
// The two states must not generate vacuum.
inline void exactRiemann(
    double x, double t, double gamma,
    double rhoL, double uL, double pL,
    double rhoR, double uR, double pR,
    double* rho, double* u, double* p // OUT: Density, velocity and pressure at (x, t)
)
{
    double cL = sqrt(gamma*pL/rhoL);
    double cR = sqrt(gamma*pR/rhoR);

    // Star pressure (Newton)
    double pStar = 0.5*(pL + pR);
    double fL, dfL, fR, dfR;
    for (int it = 0; it < 50; ++it) {
        riemannPressureFunction(pStar, gamma, rhoL, pL, cL, &fL, &dfL);
        riemannPressureFunction(pStar, gamma, rhoR, pR, cR, &fR, &dfR);
        double dp = (fL + fR + uR - uL)/(dfL + dfR);
        pStar -= dp;
        if (pStar < 1e-14) pStar = 1e-14; // Newton must not leave the positive pressures
        if (fabs(dp) < 1e-14*pStar) break;
    }
    riemannPressureFunction(pStar, gamma, rhoL, pL, cL, &fL, &dfL);
    riemannPressureFunction(pStar, gamma, rhoR, pR, cR, &fR, &dfR);
    double uStar = 0.5*(uL + uR + fR - fL);

    // Sample the waves at xi = x/t, on the corresponding side of the contact discontinuity
    double xi = x/t;
    if (xi <= uStar) {
        riemannSampleLeftSide(xi, gamma, rhoL, uL, pL, cL, pStar, uStar, rho, u, p);
    } else {
        riemannSampleLeftSide(-xi, gamma, rhoR, -uR, pR, cR, pStar, -uStar, rho, u, p);
        *u = -*u;
    }
}

#endif
