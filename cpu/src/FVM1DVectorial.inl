// Implementation of FVM1DVectorial. Not compiled on its own: included at the end of include/FVM1DVectorial.hpp

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

template <PhysicsType Physics, class NumFlux>
void FVM1DVectorial(int N, const double* __restrict__ u_0, const typename Physics::Params& params, double xL, double xR,
                    double dt, int T, BCType bc, int recordEvery, double* __restrict__ u)
{    
    const int    M   = Physics::NVARS;
    const ptrdiff_t row = ROW_SIZE(N); // We use ptrdiff because we'll be accesing the -1 index, and size_t is unsigned, so negatives are unsupported (technicallitiy)

    // Start by computing the spatial step size
    double dx = (xR - xL)/(N+1);

    // Pre compute the quotient of delta t over delta x
    double lambda = dt/dx;

    // M rows (ghost cells included): the old state uo (t^n) and the new state un (t^{n+1}).
    // uo and un point to cell 0 of their row, so uo[-1] and uo[N+1] are its ghost cells (same for un)
    double* block = (double*)malloc(sizeof(double) * 2 * STATE_SIZE(N, M));
    if (block == NULL) { fprintf(stderr, "FVM1DVectorial: not enough memory\n"); exit(1); }
    double* uo = block + 1; // + 1 so it points at the first non-ghost cell
    double* un = block + 1 + STATE_SIZE(N,M);

    // Let's copy the initial state into uo and into record 0 of the history.
    // Tricky: uo (and u) values are N+3-apart, while u_0 are N+1 apart (they don't include ghosts)
    for (int k = 0; k < M; ++k) { // Copy each variable
        memcpy(uo + k*row, u_0 + k*(N+1), sizeof(double)*(N+1));
        memcpy(u + IDX(0, k, 0, N, M), u_0 + k*(N+1), sizeof(double)*(N+1));
    }

    // Main loop
    int c_j; // Variable for iterating in space
    int t_n; // Variable for iterating in time
    for(t_n = 0; t_n < T - 1; ++t_n) { // Row 0 is the initial state, so T rows need T-1 steps
        // Ghost cells of the current row, filled in place
        fillGhosts(uo, N, M, bc);

        // Compute the values of the flux and the max speed at the left ghost cell and the first cell (for each variable).
        // We'll perform a single pass over the cells: each interface flux is computed once (per variable), 
        // and carried to the next cell.
        double UL[M], FL[M]; // UL and UR are just throw-away variables for storing the values of uo at the sides of the cell c_j
        double UR[M], FR[M];
        for (int k = 0; k < M; ++k) { UL[k] = uo[k*row - 1]; UR[k] = uo[k*row]; }
        Physics::flux(UL, FL, params); // Compute the flux at the sides of the cell c_j for all variables
        Physics::flux(UR, FR, params);
        double sL = Physics::maxSpeed(UL, params);
        double sR = Physics::maxSpeed(UR, params);

        double halfSpeed = NumFlux::halfSpeed(sL, sR); // The same for ALL variables

        // Numerical flux computation
        double numFL[M]; // F_{j-1/2}
        for (int k = 0; k < M; ++k) { numFL[k] = NumFlux::F(UL[k], UR[k], FL[k], FR[k], halfSpeed); }
        
        for (c_j = 0; c_j <= N; ++c_j) {
            // Slide: cell c_j was the right cell of the previous wall, so its values, flux and speed are already known
            for (int k = 0; k < M; ++k) { UL[k] = UR[k]; FL[k] = FR[k]; }
            sL = sR;

            // Only the new cell, c_j+1, is read and evaluated (physics once per cell)
            for (int k = 0; k < M; ++k) { UR[k] = uo[k*row + c_j + 1]; }
            Physics::flux(UR, FR, params);
            sR = Physics::maxSpeed(UR, params);
            
            halfSpeed = NumFlux::halfSpeed(sL, sR);

            double numFR[M]; // F_{j+1/2}
            for (int k = 0; k < M; ++k) { numFR[k] = NumFlux::F(UL[k], UR[k], FL[k], FR[k], halfSpeed); }
            
            for (int k = 0; k < M; ++k) { un[k*row + c_j] = uo[k*row + c_j] - lambda*(numFR[k] - numFL[k]); } // FVM Update law
            for (int k = 0; k < M; ++k) { numFL[k] = numFR[k]; } // The right wall of cell j is the left wall of cell j+1
        }

        // Copy the new state to the history if this time step is recorded
        // un and the history have the same layout (M rows of N+3, ghosts included), so the whole state is one block:
        // a single copy starting at the first ghost cell (-1). The ghost slots are copied along but never read
        if (isRecorded(t_n + 1, T, recordEvery)) { // + 1 because we just computed step t_n + 1 (uo is step t_n)
            memcpy(u + IDX(recordRow(t_n + 1, recordEvery), 0, -1, N, M), un - 1, sizeof(double)*STATE_SIZE(N, M));
        }

        // Swap the states: the new state becomes the old one for the next step,
        // and the old state (no longer needed) will be overwritten by the next new state
        double* tmp = uo;
        uo = un;
        un = tmp;
    }

    free(block);
}
