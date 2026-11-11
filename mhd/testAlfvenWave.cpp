// Test: circularly polarized Alfven wave for the 1D ideal MHD equations, periodic BCs, Rusanov flux.
// Uniform rho, P and Bx; the transverse field rotates along x (a helix) and the velocity follows it:
//     B_perp = A (sin(2 pi x/L), cos(2 pi x/L)),   v_perp = -B_perp/sqrt(rho)   (Walen relation: moves towards +x)
// Because |B_perp| is uniform, the magnetic pressure is uniform too, and this is an exact solution for any amplitude A:
// the helix travels at the Alfven speed Bx/sqrt(rho) without changing. Here it crosses the domain T_FINAL times.
// The numerical diffusion of Rusanov damps it: the test prints how much amplitude is left, and the L1 error of By.
//
// Periodic boundaries: nothing enters or leaves the domain, so the test checks:
//   - the effective CFL, which must not exceed CFL;
//   - positivity: density and pressure must stay > 0 (and not NaN);
//   - conservation: the total of each variable must not change (up to rounding).
// It returns 1 if a check fails. It records NUM_FRAMES states (plus the initial one) for a 3D animation.

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "include/FVM1DVectorial.hpp"
#include "include/HistoryBinary.h"
#include "include/CFL.hpp"
#include "MHD.hpp"

// ---------------- Simulation parameters ----------------
// The number of cells can be changed from the command line: ./testAlfvenWave [cells]
const double GAMMA       = 5.0/3.0; // Adiabatic constant
const double RHO         = 1.0;     // Uniform density
const double PRESSURE    = 1.0;     // Uniform pressure
const double BX          = 1.0;     // Field along x (Alfven speed BX/sqrt(RHO) = 1)
const double AMPLITUDE   = 1.0;     // |B_perp|: the field lines make 45 degrees with the x axis
const double xL          = 0.0;     // Left interval extrema
const double xR          = 1.0;     // Right interval extrema (one wavelength)
int          N           = 511;     // N+1 = 512 finite volumes
const double CFL         = 0.8;     // SPEED_BOUND dt / dx. dt is computed from it
const double SPEED_BOUND = 1.8;     // Upper bound of |vx| + c_f (exact solution: c_f = 1.77 everywhere, vx = 0)
const double T_FINAL     = 5.0;     // Final time: 5 crossings of the domain
const int    NUM_FRAMES  = 450;     // Records written after the initial state (at most): 15 s at 30 frames/s
const char*  OUTPUT_FILE = "alfven_wave.bin";
// -------------------------------------------------------

const int M = MHD::NVARS;

// Primitive variables (rho, vx, vy, vz, Bx, By, Bz, P) of the wave at position x and time t (exact solution)
void exactSolution(double x, double t, double* W)
{
    double phase = 2.0*M_PI*(x - t*BX/sqrt(RHO))/(xR - xL);
    double By = AMPLITUDE*sin(phase), Bz = AMPLITUDE*cos(phase);
    W[0] = RHO; W[1] = 0.0; W[2] = -By/sqrt(RHO); W[3] = -Bz/sqrt(RHO);
    W[4] = BX;  W[5] = By;  W[6] = Bz;            W[7] = PRESSURE;
}

void initialCondition(double x, double* W) { exactSolution(x, 0.0, W); }

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
    FVM1DVectorial<MHD>(N, u0, params, xL, xR, dt, T, BC_PERIODIC, recordEvery, u);
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

    // Conservation: with periodic boundaries the total of each variable must not change (largest relative error)
    double conservationError = 0.0;
    for (int k = 0; k < M; ++k) {
        double sum0 = 0.0, sum1 = 0.0, scale = 0.0;
        for (int j = 0; j <= N; ++j) {
            double a = u0[k*(N+1) + j], b = u[IDX(R-1, k, j, N, M)];
            sum0 += a; sum1 += b; scale += fabs(a) + fabs(b);
        }
        if (scale > 0.0) conservationError = fmax(conservationError, fabs(sum1 - sum0)/scale);
    }

    double cfl = effectiveCFL<MHD>(u, N, T, recordEvery, params, dt, dx);
    printf("Cells: %d, time steps: %d, record every: %d (%d records)\n", N+1, T, recordEvery, R);
    printf("dt = %g, final time = %g\n", dt, dt*(T-1));
    printf("Elapsed time: %.3f ms\n", 1e3*elapsed);
    printf("Effective CFL: %.3f (target %.2f)\n", cfl, CFL);
    printf("Min density: %.6e, min pressure: %.6e, cells with rho <= 0 or P <= 0: %d\n", minRho, minP, badCells);
    printf("Conservation error: %.3e (relative, largest over the %d variables)\n", conservationError, M);

    // Damping and accuracy: amplitude left (largest |B_perp|) and relative L1 error of By against the exact solution
    double maxBperp = 0.0, errSum = 0.0, exSum = 0.0;
    for (int j = 0; j <= N; ++j) {
        double x = xL + (j + 0.5)*dx, We[M];
        exactSolution(x, T_FINAL, We);
        double By = u[IDX(R-1, 5, j, N, M)], Bz = u[IDX(R-1, 6, j, N, M)];
        maxBperp = fmax(maxBperp, sqrt(By*By + Bz*Bz));
        errSum += fabs(By - We[5]);
        exSum  += fabs(We[5]);
    }
    printf("Amplitude left: %.3f of the initial one (numerical diffusion)\n", maxBperp/AMPLITUDE);
    printf("Relative L1 error of By at t = %g: %.6e\n", T_FINAL, errSum/exSum);

    int passed = (cfl <= CFL + 1e-12) && (badCells == 0) && (conservationError < 1e-10);
    printf("Checks: %s\n", passed ? "passed" : "FAILED");

    writeHistoryBinary(OUTPUT_FILE, u, N, M, T, xL, xR, dt, recordEvery);

    free(u0);
    free(u);
    return passed ? 0 : 1;
}
