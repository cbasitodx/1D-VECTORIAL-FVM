// Test: two interacting blast waves in a periodic tube, 1D Euler equations, Rusanov flux.
// Like the Woodward-Colella blast waves, but with periodic boundaries instead of reflecting walls:
// gas at rest with rho = 1 everywhere, and two hot slabs at very high pressure in a cold background:
//     p = 1000 in [0.1, 0.2],   p = 100 in [0.6, 0.7],   p = 0.01 elsewhere      (gamma = 1.4)
// Both slabs explode: strong shocks sweep the gas into thin dense shells, which collide (in the middle and across the
// periodic boundary), giving reflected shocks, contact discontinuities and rarefactions.
// It records up to NUM_RECORDS states (many more than a video needs) so that the tracer particles of the animation
// (PYTHON_PROTOTYPES/euler) can be integrated accurately from the recorded velocities. The test checks:
//   - the effective CFL, which must not exceed CFL;
//   - positivity at every recorded state: density and pressure must stay > 0 (and not NaN);
//   - conservation: the total mass, momentum and energy must not change (up to rounding).
// It returns 1 if a check fails.

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "include/FVM1DVectorial.hpp"
#include "include/HistoryBinary.h"
#include "include/CFL.hpp"
#include "Euler.hpp"

// ---------------- Simulation parameters ----------------
// The number of cells can be changed from the command line: ./testEulerBlastPeriodic [cells]
const double GAMMA       = 1.4;    // Ratio of specific heats
const double xL          = 0.0;    // Left interval extrema
const double xR          = 1.0;    // Right interval extrema
int          N           = 799;    // N+1 = 800 finite volumes
const double CFL         = 0.8;    // SPEED_BOUND dt / dx. dt is computed from it
const double SPEED_BOUND = 56.0;   // Upper bound of |u| + c during the run (check the printed effective CFL)
const double T_FINAL     = 0.05;   // Final time: the explosions, the collisions and the first reflected waves
const int    NUM_RECORDS = 2000;   // Records written after the initial state (at most)
const char*  OUTPUT_FILE = "euler_blast_periodic.bin";
// -------------------------------------------------------

const int M = Euler::NVARS;

// Pressure of the initial condition at position x (the gas is at rest with rho = 1 everywhere)
double initialPressure(double x)
{
    if (x >= 0.1 && x < 0.2) return 1000.0; // Strong blast
    if (x >= 0.6 && x < 0.7) return 100.0;  // Weaker blast
    return 0.01;                            // Cold background
}

int main(int argc, char** argv)
{
    // Optional number of cells from the command line (the value above otherwise)
    if (argc == 2) N = atoi(argv[1]) - 1;
    else if (argc != 1) { fprintf(stderr, "Usage: %s [cells]\n", argv[0]); return 1; }

    double dx = (xR - xL)/(N+1);
    Euler::Params params = { GAMMA };

    // Fixed time step: the fewest steps that keep SPEED_BOUND dt/dx <= CFL and end exactly at T_FINAL
    int    steps       = (int)ceil(T_FINAL*SPEED_BOUND/(CFL*dx));
    double dt          = T_FINAL/steps;
    int    T           = steps + 1;                             // Time steps, initial state included
    int    recordEvery = (steps + NUM_RECORDS - 1)/NUM_RECORDS; // At most NUM_RECORDS records after the initial one

    // Initial condition, evaluated at the cell centers: (rho, rho u, E) with E = p/(gamma - 1) + rho u^2/2
    int     R  = numRecords(T, recordEvery);
    double* u0 = (double*)malloc(sizeof(double)*M*(N+1));
    double* u  = (double*)malloc(sizeof(double)*HISTORY_SIZE(R, N, M));
    if (u0 == NULL || u == NULL) { fprintf(stderr, "Not enough memory\n"); return 1; }
    for (int j = 0; j <= N; ++j) {
        u0[0*(N+1) + j] = 1.0;
        u0[1*(N+1) + j] = 0.0;
        u0[2*(N+1) + j] = initialPressure(xL + (j + 0.5)*dx)/(GAMMA - 1.0);
    }

    // Solve and time the whole history computation
    struct timespec t0, t1;
    timespec_get(&t0, TIME_UTC);
    FVM1DVectorial<Euler>(N, u0, params, xL, xR, dt, T, BC_PERIODIC, recordEvery, u);
    timespec_get(&t1, TIME_UTC);
    double elapsed = (t1.tv_sec - t0.tv_sec) + 1e-9*(t1.tv_nsec - t0.tv_nsec);

    // Positivity at every recorded state. "!(x > 0)" also catches NaN
    double minRho = INFINITY, minP = INFINITY, maxRho = 0.0;
    int    badCells = 0;
    for (int r = 0; r < R; ++r) {
        for (int j = 0; j <= N; ++j) {
            double rho = u[IDX(r, 0, j, N, M)], mom = u[IDX(r, 1, j, N, M)], E = u[IDX(r, 2, j, N, M)];
            double p = (GAMMA - 1.0)*(E - 0.5*mom*mom/rho);
            if (!(rho > 0.0) || !(p > 0.0)) { ++badCells; continue; }
            minRho = fmin(minRho, rho);
            maxRho = fmax(maxRho, rho);
            minP   = fmin(minP, p);
        }
    }

    // Conservation: with periodic boundaries, the total mass, momentum and energy must not change
    double conservationError = 0.0;
    for (int k = 0; k < M; ++k) {
        double sum0 = 0.0, sum1 = 0.0, scale = 0.0;
        for (int j = 0; j <= N; ++j) {
            double a = u0[k*(N+1) + j], b = u[IDX(R-1, k, j, N, M)];
            sum0 += a; sum1 += b; scale += fabs(a) + fabs(b);
        }
        if (scale > 0.0) conservationError = fmax(conservationError, fabs(sum1 - sum0)/scale);
    }

    double cfl = effectiveCFL<Euler>(u, N, T, recordEvery, params, dt, dx);
    printf("Cells: %d, time steps: %d, record every: %d (%d records)\n", N+1, T, recordEvery, R);
    printf("dt = %g, final time = %g\n", dt, dt*(T-1));
    printf("Elapsed time: %.3f ms\n", 1e3*elapsed);
    printf("Effective CFL: %.3f (target %.2f)\n", cfl, CFL);
    printf("Density over the run: [%.4e, %.4e], min pressure: %.4e, cells with rho <= 0 or p <= 0: %d\n",
           minRho, maxRho, minP, badCells);
    printf("Conservation error: %.3e (relative, largest of mass, momentum and energy)\n", conservationError);

    int passed = (cfl <= CFL + 1e-12) && (badCells == 0) && (conservationError < 1e-10);
    printf("Checks: %s\n", passed ? "passed" : "FAILED");

    writeHistoryBinary(OUTPUT_FILE, u, N, M, T, xL, xR, dt, recordEvery);

    free(u0);
    free(u);
    return passed ? 0 : 1;
}
