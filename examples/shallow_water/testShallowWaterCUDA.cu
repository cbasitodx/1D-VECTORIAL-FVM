// CUDA test: dam break for the shallow water equations, transmissive BCs, Rusanov flux.
// Same setup as testShallowWater.cpp, so the results can be compared with the CPU version.

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>
#include "include/FVM1DVectorialCUDA.cuh"
#include "include/HistoryBinary.h"
#include "include/CFL.hpp"
#include "ShallowWater.hpp"
#include "ExactDamBreak.hpp"

// ---------------- Simulation parameters ----------------
// N, T and RECORD_EVERY can be changed from the command line: ./test [N T RECORD_EVERY] (benchmark.sh does)
const double g                 = 9.81;   // Gravity
const double HL                = 2.0;    // Depth left of the dam
const double HR                = 1.0;    // Depth right of the dam
const double xL                = -1.0;   // Left interval extrema
const double xR                =  1.0;   // Right interval extrema
int          N                 = 999999; // N+1 = 10^6 finite volumes
const double CFL               = 0.9;    // SPEED_BOUND dt / dx. dt is computed from it
const double SPEED_BOUND       = 6.0;    // Upper bound of |u| + sqrt(g h) during the run (check the printed effective CFL)
int          T                 = 2001;   // Time steps (final time = dt*(T-1))
int          RECORD_EVERY      = 100;    // Copy back and write every RECORD_EVERY time steps (first and last always)
const char*  OUTPUT_FILE       = "shallow_water_cuda.bin";
const int    THREADS_PER_BLOCK = 256;    // Threads per block
const int    NUM_BLOCKS        = 3907;   // Blocks: 3907*256 >= 10^6 threads, one per cell. Any value works (grid-stride loop):
                                         // with fewer threads than cells, each thread updates several cells
// -------------------------------------------------------

const int M = ShallowWater::NVARS;

// Writes the conserved variables (h, hu) at position x
void initialCondition(double x, double* U)
{
    U[0] = (x < 0.0) ? HL : HR;
    U[1] = 0.0;
}

int main(int argc, char** argv)
{
    // Optional sizes from the command line (the values above otherwise)
    if (argc == 4) { N = atoi(argv[1]); T = atoi(argv[2]); RECORD_EVERY = atoi(argv[3]); }
    else if (argc != 1) { fprintf(stderr, "Usage: %s [N T RECORD_EVERY]\n", argv[0]); return 1; }

    double dx = (xR - xL)/(N+1);
    ShallowWater::Params params = { g };

    // Initial condition, evaluated at the cell centers
    int     R  = numRecords(T, RECORD_EVERY);
    double* u0 = (double*)malloc(sizeof(double)*M*(N+1));
    double* u  = (double*)malloc(sizeof(double)*HISTORY_SIZE(R, N, M));
    if (u0 == NULL || u == NULL) { fprintf(stderr, "Not enough memory\n"); return 1; }
    for (int j = 0; j <= N; ++j) {
        double U[M] = {};
        initialCondition(xL + (j + 0.5)*dx, U);
        for (int k = 0; k < M; ++k) u0[k*(N+1) + j] = U[k];
    }

    // Fixed time step from the speed bound. For systems, the maximum speed of u0 is not a bound for the whole run
    double dt     = CFL*dx/SPEED_BOUND;
    double tFinal = dt*(T-1);

    // Create the CUDA context before timing (it takes ~100 ms and is not part of the computation)
    cudaFree(0);

    // Solve and time the whole history computation (end to end: GPU memory, copies and kernels)
    double kernelSeconds;
    struct timespec t0, t1;
    timespec_get(&t0, TIME_UTC);
    FVM1DVectorialCUDA<ShallowWater>(N, u0, params, xL, xR, dt, T, BC_TRANSMISSIVE, RECORD_EVERY, u,
                                     THREADS_PER_BLOCK, NUM_BLOCKS, &kernelSeconds);
    timespec_get(&t1, TIME_UTC);
    double elapsed = (t1.tv_sec - t0.tv_sec) + 1e-9*(t1.tv_nsec - t0.tv_nsec);

    // Relative L1 error of h at the final time
    double errSum = 0.0, exSum = 0.0;
    for (int j = 0; j <= N; ++j) {
        double hExact, uExact;
        exactDamBreak(xL + (j + 0.5)*dx, tFinal, g, HL, HR, &hExact, &uExact);
        errSum += fabs(u[IDX(R-1, 0, j, N, M)] - hExact);
        exSum  += fabs(hExact);
    }

    printf("Cells: %d, time steps: %d, record every: %d\n", N+1, T, RECORD_EVERY);
    printf("dt = %g, final time = %g\n", dt, tFinal);
    printf("Elapsed time: %.3f ms (kernels only: %.3f ms)\n", 1e3*elapsed, 1e3*kernelSeconds);
    printf("Time per cell update: %.3f ns (kernels only: %.3f ns)\n", 1e9*elapsed/((double)(T-1)*(N+1)), 1e9*kernelSeconds/((double)(T-1)*(N+1)));
    printf("Effective CFL: %.3f\n", effectiveCFL<ShallowWater>(u, N, T, RECORD_EVERY, params, dt, dx));
    printf("Relative L1 error of h at t = %g: %.6e\n", tFinal, errSum/exSum);

    writeHistoryBinary(OUTPUT_FILE, u, N, M, T, xL, xR, dt, RECORD_EVERY);

    free(u0);
    free(u);
    return 0;
}
