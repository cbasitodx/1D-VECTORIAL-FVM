// CUDA test: Burgers equation u_t + (u^2/2)_x = 0, periodic BCs, Rusanov flux.
// Same setup as testBurgers.cpp, so the results can be compared with the CPU version.

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>
#include "include/FVM1DVectorialCUDA.cuh"
#include "include/HistoryBinary.h"
#include "Burgers.hpp"

// ---------------- Simulation parameters ----------------
// N, T and RECORD_EVERY can be changed from the command line: ./test [N T RECORD_EVERY] (benchmark.sh does)
const double xL                = -1.0;   // Left interval extrema
const double xR                =  1.0;   // Right interval extrema
int          N                 = 999999; // N+1 = 10^6 finite volumes
const double CFL               = 1.0;    // max|f'(u)| dt / dx (stable for CFL <= 1). dt is computed from it
int          T                 = 2001;   // Time steps (final time = dt*(T-1))
int          RECORD_EVERY      = 100;    // Copy back and write every RECORD_EVERY time steps (first and last always)
const char*  OUTPUT_FILE       = "burgers_cuda.bin";
const int    THREADS_PER_BLOCK = 256;    // Threads per block
const int    NUM_BLOCKS        = 3907;   // Blocks: 3907*256 >= 10^6 threads, one per cell. Any value works (grid-stride loop):
                                         // with fewer threads than cells, each thread updates several cells
// -------------------------------------------------------

const int M = Burgers::NVARS;

double initialCondition(double x) { return (x < 0.0) ? 1.0 : 0.0; }

int main(int argc, char** argv)
{
    // Optional sizes from the command line (the values above otherwise)
    if (argc == 4) { N = atoi(argv[1]); T = atoi(argv[2]); RECORD_EVERY = atoi(argv[3]); }
    else if (argc != 1) { fprintf(stderr, "Usage: %s [N T RECORD_EVERY]\n", argv[0]); return 1; }

    double dx = (xR - xL)/(N+1);
    Burgers::Params params = {};

    // Initial condition, evaluated at the cell centers
    int     R  = numRecords(T, RECORD_EVERY);
    double* u0 = (double*)malloc(sizeof(double)*M*(N+1));
    double* u  = (double*)malloc(sizeof(double)*HISTORY_SIZE(R, N, M));
    if (u0 == NULL || u == NULL) { fprintf(stderr, "Not enough memory\n"); return 1; }
    for (int j = 0; j <= N; ++j) u0[j] = initialCondition(xL + (j + 0.5)*dx);

    // Time step from the CFL number. For a scalar law the scheme keeps u within the initial range,
    // so the maximum speed of u0 bounds the wave speed for the whole run
    double maxSpeed = 0.0;
    for (int j = 0; j <= N; ++j) maxSpeed = fmax(maxSpeed, Burgers::maxSpeed(&u0[j], params));
    double dt = CFL*dx/maxSpeed;

    // Create the CUDA context before timing (it takes ~100 ms and is not part of the computation)
    cudaFree(0);

    // Solve and time the whole history computation (end to end: GPU memory, copies and kernels)
    double kernelSeconds;
    struct timespec t0, t1;
    timespec_get(&t0, TIME_UTC);
    FVM1DVectorialCUDA<Burgers>(N, u0, params, xL, xR, dt, T, BC_PERIODIC, RECORD_EVERY, u,
                                THREADS_PER_BLOCK, NUM_BLOCKS, &kernelSeconds);
    timespec_get(&t1, TIME_UTC);
    double elapsed = (t1.tv_sec - t0.tv_sec) + 1e-9*(t1.tv_nsec - t0.tv_nsec);

    printf("Cells: %d, time steps: %d, record every: %d\n", N+1, T, RECORD_EVERY);
    printf("dt = %g, final time = %g\n", dt, dt*(T-1));
    printf("Elapsed time: %.3f ms (kernels only: %.3f ms)\n", 1e3*elapsed, 1e3*kernelSeconds);
    printf("Time per cell update: %.3f ns (kernels only: %.3f ns)\n", 1e9*elapsed/((double)(T-1)*(N+1)), 1e9*kernelSeconds/((double)(T-1)*(N+1)));

    writeHistoryBinary(OUTPUT_FILE, u, N, M, T, xL, xR, dt, RECORD_EVERY);

    free(u0);
    free(u);
    return 0;
}
