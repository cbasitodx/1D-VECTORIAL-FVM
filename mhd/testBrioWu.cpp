// Test: Brio-Wu shock tube for the 1D ideal MHD equations, transmissive BCs, Rusanov flux.
// Setup from Castro-Díaz et al., "A second order PVM flux limiter method" (2014), Section 4.1:
//     (rho, vx, vy, vz, Bx, By, Bz, P) = (1,     0, 0, 0, 0.75,  1, 0, 1)    if x <= 0
//                                        (0.125, 0, 0, 0, 0.75, -1, 0, 0.1)  if x >  0
//     gamma = 2, domain [-1, 1], 800 cells, CFL 0.8, final time 0.2.
//
// The solution has no closed form (the paper compares with a fine-mesh reference), so the test checks:
//   - the effective CFL, which must not exceed CFL;
//   - positivity: density and pressure must stay > 0 (and not NaN);
//   - conservation: until the waves reach the boundaries, the boundary cells keep their initial state, so the total
//     of each variable changes exactly by the boundary fluxes: sum_j U_j(t) = sum_j U_j(0) - (t/dx)*(f(U_right) - f(U_left)).
//     If the boundary cells did change (the numerical diffusion of Rusanov can spread a wave up to the boundary),
//     this check does not apply and is skipped.
// It returns 1 if a check fails.

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "include/FVM1DVectorial.hpp"
#include "include/HistoryBinary.h"
#include "include/CFL.hpp"
#include "MHD.hpp"

// ---------------- Simulation parameters ----------------
// The number of cells can be changed from the command line: ./testBrioWu [cells]
const double GAMMA       = 2.0;    // Adiabatic constant
const double xL          = -1.0;   // Left interval extrema
const double xR          =  1.0;   // Right interval extrema
int          N           = 799;    // N+1 = 800 finite volumes (as in the paper)
const double CFL         = 0.8;    // SPEED_BOUND dt / dx. dt is computed from it
const double SPEED_BOUND = 3.9;    // Upper bound of |vx| + c_f during the run (check the printed effective CFL)
const double T_FINAL     = 0.2;    // Final time
const int    NUM_FRAMES  = 100;    // Records written after the initial state (at most), for the animation
const char*  OUTPUT_FILE = "brio_wu.bin";
// -------------------------------------------------------

const int M = MHD::NVARS;

// Primitive variables (rho, vx, vy, vz, Bx, By, Bz, P) of the initial condition at position x
void initialCondition(double x, double* W)
{
    double left[M]  = { 1.0,   0.0, 0.0, 0.0, 0.75,  1.0, 0.0, 1.0 };
    double right[M] = { 0.125, 0.0, 0.0, 0.0, 0.75, -1.0, 0.0, 0.1 };
    for (int k = 0; k < M; ++k) W[k] = (x <= 0.0) ? left[k] : right[k];
}

int main(int argc, char** argv)
{
    // Optional number of cells from the command line (the value above otherwise)
    if (argc == 2) N = atoi(argv[1]) - 1;
    else if (argc != 1) { fprintf(stderr, "Usage: %s [cells]\n", argv[0]); return 1; }

    double dx = (xR - xL)/(N+1);
    MHD::Params params = { GAMMA };

    // Fixed time step: the fewest steps that keep SPEED_BOUND dt/dx <= CFL and end exactly at T_FINAL
    int    steps       = (int)ceil(T_FINAL*SPEED_BOUND/(CFL*dx));
    double dt          = T_FINAL/steps;
    int    T           = steps + 1;                           // Time steps, initial state included
    int    recordEvery = (steps + NUM_FRAMES - 1)/NUM_FRAMES; // At most NUM_FRAMES records after the initial one

    // Initial condition, evaluated at the cell centers and converted to conserved variables
    int     R  = numRecords(T, recordEvery);
    double* u0 = (double*)malloc(sizeof(double)*M*(N+1));
    double* u  = (double*)malloc(sizeof(double)*HISTORY_SIZE(R, N, M));
    if (u0 == NULL || u == NULL) { fprintf(stderr, "Not enough memory\n"); return 1; }
    for (int j = 0; j <= N; ++j) {
        double W[M], U[M];
        initialCondition(xL + (j + 0.5)*dx, W);
        MHD::primitiveToConserved(W, U, params);
        for (int k = 0; k < M; ++k) u0[k*(N+1) + j] = U[k];
    }

    // Solve and time the whole history computation
    struct timespec t0, t1;
    timespec_get(&t0, TIME_UTC);
    FVM1DVectorial<MHD>(N, u0, params, xL, xR, dt, T, BC_TRANSMISSIVE, recordEvery, u);
    timespec_get(&t1, TIME_UTC);
    double elapsed = (t1.tv_sec - t0.tv_sec) + 1e-9*(t1.tv_nsec - t0.tv_nsec);

    // Positivity at the final time. "!(x > 0)" also catches NaN
    double minRho = INFINITY, minP = INFINITY;
    int    badCells = 0;
    for (int j = 0; j <= N; ++j) {
        double U[M], W[M];
        for (int k = 0; k < M; ++k) U[k] = u[IDX(R-1, k, j, N, M)];
        MHD::conservedToPrimitive(U, W, params);
        minRho = fmin(minRho, W[0]);
        minP   = fmin(minP, W[7]);
        if (!(W[0] > 0.0) || !(W[7] > 0.0)) ++badCells;
    }

    // Conservation (see the top of the file): largest relative error over the M variables
    double UL[M], UR[M], fL[M], fR[M];
    for (int k = 0; k < M; ++k) { UL[k] = u0[k*(N+1)]; UR[k] = u0[k*(N+1) + N]; }
    MHD::flux(UL, fL, params);
    MHD::flux(UR, fR, params);
    double conservationError = 0.0, boundaryChange = 0.0;
    for (int k = 0; k < M; ++k) {
        // The check is only valid if the boundary cells kept their initial state (the waves did not reach them)
        double change = fmax(fabs(u[IDX(R-1, k, 0, N, M)] - UL[k]), fabs(u[IDX(R-1, k, N, N, M)] - UR[k]));
        boundaryChange = fmax(boundaryChange, change/fmax(1.0, fmax(fabs(UL[k]), fabs(UR[k]))));
        double sum0 = 0.0, sum1 = 0.0, scale = 0.0;
        for (int j = 0; j <= N; ++j) {
            double a = u0[k*(N+1) + j], b = u[IDX(R-1, k, j, N, M)];
            sum0 += a; sum1 += b; scale += fabs(a) + fabs(b);
        }
        double expected = sum0 - (T_FINAL/dx)*(fR[k] - fL[k]);
        if (scale > 0.0) conservationError = fmax(conservationError, fabs(sum1 - expected)/scale);
    }

    double cfl = effectiveCFL<MHD>(u, N, T, recordEvery, params, dt, dx);
    printf("Cells: %d, time steps: %d, record every: %d\n", N+1, T, recordEvery);
    printf("dt = %g, final time = %g\n", dt, dt*(T-1));
    printf("Elapsed time: %.3f ms\n", 1e3*elapsed);
    printf("Effective CFL: %.3f (target %.2f)\n", cfl, CFL);
    printf("Min density: %.6e, min pressure: %.6e, cells with rho <= 0 or P <= 0: %d\n", minRho, minP, badCells);
    int conservationChecked = boundaryChange < 1e-12;
    if (conservationChecked)
        printf("Conservation error: %.3e (relative, largest over the %d variables)\n", conservationError, M);
    else
        printf("Conservation: not checked, the waves reached the boundary (boundary cells changed by up to %.2e)\n",
               boundaryChange);

    int passed = (cfl <= CFL + 1e-12) && (badCells == 0) && (!conservationChecked || conservationError < 1e-10);
    printf("Checks: %s\n", passed ? "passed" : "FAILED");

    writeHistoryBinary(OUTPUT_FILE, u, N, M, T, xL, xR, dt, recordEvery);

    free(u0);
    free(u);
    return passed ? 0 : 1;
}
