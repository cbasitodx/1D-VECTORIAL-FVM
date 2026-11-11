// Test: linear transport equation u_t + a u_x = 0, periodic BCs, Rusanov flux.
// Initial condition: u = 1 if x < 0, u = 0 if x > 0.
// Exact solution: u(x, t) = u_0(x - a t), wrapped periodically into [xL, xR).

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "include/FVM1DVectorial.hpp"
#include "include/HistoryBinary.h"
#include "LinearTransport.hpp"

// ---------------- Simulation parameters ----------------
// N, T and RECORD_EVERY can be changed from the command line: ./test [N T RECORD_EVERY] (benchmark.sh does)
const double a            = 1.0;    // Transport velocity
const double xL           = -1.0;   // Left interval extrema
const double xR           =  1.0;   // Right interval extrema
int          N            = 999999; // N+1 = 10^6 finite volumes
const double CFL          = 1.0;    // max|f'(u)| dt / dx (stable for CFL <= 1). dt is computed from it
int          T            = 2001;   // Time steps (final time = dt*(T-1))
int          RECORD_EVERY = 100;    // Store and write every RECORD_EVERY time steps (first and last always)
const char*  OUTPUT_FILE  = "linear_transport.bin";
// -------------------------------------------------------

const int M = LinearTransport::NVARS;

double initialCondition(double x) { return (x < 0.0) ? 1.0 : 0.0; }

double exactSolution(double x, double t)
{
    double L  = xR - xL;
    double xi = x - a*t;
    xi = xi - L*floor((xi - xL)/L); // Wrap back into [xL, xR)
    return initialCondition(xi);
}

int main(int argc, char** argv)
{
    // Optional sizes from the command line (the values above otherwise)
    if (argc == 4) { N = atoi(argv[1]); T = atoi(argv[2]); RECORD_EVERY = atoi(argv[3]); }
    else if (argc != 1) { fprintf(stderr, "Usage: %s [N T RECORD_EVERY]\n", argv[0]); return 1; }

    double dx = (xR - xL)/(N+1);
    LinearTransport::Params params = { a };

    // Initial condition, evaluated at the cell centers
    int     R  = numRecords(T, RECORD_EVERY);
    double* u0 = (double*)malloc(sizeof(double)*M*(N+1));
    double* u  = (double*)malloc(sizeof(double)*HISTORY_SIZE(R, N, M));
    if (u0 == NULL || u == NULL) { fprintf(stderr, "Not enough memory\n"); return 1; }
    for (int j = 0; j <= N; ++j) u0[j] = initialCondition(xL + (j + 0.5)*dx);

    // Time step from the CFL number. The wave speed is f'(u) = a everywhere
    double dt     = CFL*dx/fabs(a);
    double tFinal = dt*(T-1);

    // Solve and time the whole history computation
    struct timespec t0, t1;
    timespec_get(&t0, TIME_UTC);
    FVM1DVectorial<LinearTransport>(N, u0, params, xL, xR, dt, T, BC_PERIODIC, RECORD_EVERY, u);
    timespec_get(&t1, TIME_UTC);
    double elapsed = (t1.tv_sec - t0.tv_sec) + 1e-9*(t1.tv_nsec - t0.tv_nsec);

    // Relative L1 error at the final time
    double errSum = 0.0, exSum = 0.0;
    for (int j = 0; j <= N; ++j) {
        double ue = exactSolution(xL + (j + 0.5)*dx, tFinal);
        errSum += fabs(u[IDX(R-1, 0, j, N, M)] - ue);
        exSum  += fabs(ue);
    }

    printf("Cells: %d, time steps: %d, record every: %d\n", N+1, T, RECORD_EVERY);
    printf("dt = %g, final time = %g\n", dt, tFinal);
    printf("Elapsed time: %.3f ms\n", 1e3*elapsed);
    printf("Time per cell update: %.3f ns\n", 1e9*elapsed/((double)(T-1)*(N+1)));
    printf("Relative L1 error at t = %g: %.6e\n", tFinal, errSum/exSum);

    writeHistoryBinary(OUTPUT_FILE, u, N, M, T, xL, xR, dt, RECORD_EVERY);

    free(u0);
    free(u);
    return 0;
}
