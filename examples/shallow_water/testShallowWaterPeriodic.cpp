// Test: shallow water in a periodic channel, Rusanov flux. A long run for a 3D video (PYTHON_PROTOTYPES/shallow_water).
// Water at rest, 1 m deep, with:
//   - a tall, narrow hump at x = 4 (+1.2 m): it collapses into two steep fronts;
//   - a wide, low hump at x = 10 (+0.8 m): smoother, slower waves;
//   - a fast current around x = 15.5 (4.5 m/s, Froude number ~1.4: supercritical): it runs into the water ahead of it
//     and forms a bore (a hydraulic jump).
// With periodic boundaries the waves keep crossing the channel and colliding. The test checks:
//   - the effective CFL, which must not exceed CFL;
//   - positivity at every recorded state: the depth h must stay > 0 (and not NaN);
//   - conservation: the total mass and momentum must not change (up to rounding).
// It returns 1 if a check fails.

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "include/FVM1DVectorial.hpp"
#include "include/HistoryBinary.h"
#include "include/CFL.hpp"
#include "ShallowWater.hpp"

// ---------------- Simulation parameters ----------------
// The number of cells can be changed from the command line: ./testShallowWaterPeriodic [cells]
const double g           = 9.81;   // Gravity
const double xL          = 0.0;    // Left interval extrema
const double xR          = 20.0;   // Right interval extrema (channel length, m)
int          N           = 799;    // N+1 = 800 finite volumes
const double CFL         = 0.8;    // SPEED_BOUND dt / dx. dt is computed from it
const double SPEED_BOUND = 8.0;    // Upper bound of |u| + sqrt(g h) during the run (check the printed effective CFL)
const double T_FINAL     = 10.0;   // Final time (s)
const int    NUM_FRAMES  = 450;    // Records written after the initial state (at most): 15 s of video at 30 frames/s
const char*  OUTPUT_FILE = "shallow_water_periodic.bin";
// -------------------------------------------------------

const int M = ShallowWater::NVARS;

// Depth h and velocity u of the initial condition at position x
void initialCondition(double x, double* h, double* u)
{
    *h = 1.0 + 1.2*exp(-((x - 4.0)/0.5)*((x - 4.0)/0.5))     // Tall, narrow hump
             + 0.8*exp(-((x - 10.0)/1.2)*((x - 10.0)/1.2));  // Wide, low hump
    double s = (x - 15.5)/1.0;
    *u = 4.5*exp(-s*s*s*s);                                   // Fast current (flat-topped profile)
}

int main(int argc, char** argv)
{
    // Optional number of cells from the command line (the value above otherwise)
    if (argc == 2) N = atoi(argv[1]) - 1;
    else if (argc != 1) { fprintf(stderr, "Usage: %s [cells]\n", argv[0]); return 1; }

    double dx = (xR - xL)/(N+1);
    ShallowWater::Params params = { g };

    // Fixed time step: the fewest steps that keep SPEED_BOUND dt/dx <= CFL and end exactly at T_FINAL
    int    steps       = (int)ceil(T_FINAL*SPEED_BOUND/(CFL*dx));
    double dt          = T_FINAL/steps;
    int    T           = steps + 1;                           // Time steps, initial state included
    int    recordEvery = (steps + NUM_FRAMES - 1)/NUM_FRAMES; // At most NUM_FRAMES records after the initial one

    // Initial condition, evaluated at the cell centers: (h, hu)
    int     R  = numRecords(T, recordEvery);
    double* u0 = (double*)malloc(sizeof(double)*M*(N+1));
    double* u  = (double*)malloc(sizeof(double)*HISTORY_SIZE(R, N, M));
    if (u0 == NULL || u == NULL) { fprintf(stderr, "Not enough memory\n"); return 1; }
    for (int j = 0; j <= N; ++j) {
        double h, vel;
        initialCondition(xL + (j + 0.5)*dx, &h, &vel);
        u0[0*(N+1) + j] = h;
        u0[1*(N+1) + j] = h*vel;
    }

    // Solve and time the whole history computation
    struct timespec t0, t1;
    timespec_get(&t0, TIME_UTC);
    FVM1DVectorial<ShallowWater>(N, u0, params, xL, xR, dt, T, BC_PERIODIC, recordEvery, u);
    timespec_get(&t1, TIME_UTC);
    double elapsed = (t1.tv_sec - t0.tv_sec) + 1e-9*(t1.tv_nsec - t0.tv_nsec);

    // Positivity at every recorded state. "!(h > 0)" also catches NaN
    double minH = INFINITY, maxFroude = 0.0;
    int    badCells = 0;
    for (int r = 0; r < R; ++r) {
        for (int j = 0; j <= N; ++j) {
            double h = u[IDX(r, 0, j, N, M)], hu = u[IDX(r, 1, j, N, M)];
            if (!(h > 0.0)) { ++badCells; continue; }
            minH      = fmin(minH, h);
            maxFroude = fmax(maxFroude, fabs(hu/h)/sqrt(g*h));
        }
    }

    // Conservation: with periodic boundaries, the total mass (h) and momentum (hu) must not change
    double conservationError = 0.0;
    for (int k = 0; k < M; ++k) {
        double sum0 = 0.0, sum1 = 0.0, scale = 0.0;
        for (int j = 0; j <= N; ++j) {
            double a = u0[k*(N+1) + j], b = u[IDX(R-1, k, j, N, M)];
            sum0 += a; sum1 += b; scale += fabs(a) + fabs(b);
        }
        if (scale > 0.0) conservationError = fmax(conservationError, fabs(sum1 - sum0)/scale);
    }

    double cfl = effectiveCFL<ShallowWater>(u, N, T, recordEvery, params, dt, dx);
    printf("Cells: %d, time steps: %d, record every: %d (%d records)\n", N+1, T, recordEvery, R);
    printf("dt = %g, final time = %g\n", dt, dt*(T-1));
    printf("Elapsed time: %.3f ms\n", 1e3*elapsed);
    printf("Effective CFL: %.3f (target %.2f)\n", cfl, CFL);
    printf("Min depth over the run: %.6e, cells with h <= 0: %d\n", minH, badCells);
    printf("Max Froude number over the run: %.3f\n", maxFroude);
    printf("Conservation error: %.3e (relative, largest of mass and momentum)\n", conservationError);

    int passed = (cfl <= CFL + 1e-12) && (badCells == 0) && (conservationError < 1e-10);
    printf("Checks: %s\n", passed ? "passed" : "FAILED");

    writeHistoryBinary(OUTPUT_FILE, u, N, M, T, xL, xR, dt, recordEvery);

    free(u0);
    free(u);
    return passed ? 0 : 1;
}
