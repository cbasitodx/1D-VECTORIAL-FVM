#ifndef BOUNDARY_CONDITIONS_H
#define BOUNDARY_CONDITIONS_H

#include "include/Portability.h" // HOST_DEVICE
#include "include/Index.h"       // ROW_SIZE

// Boundary conditions supported by the solvers (shared by cpu/ and cuda/).
// The ghost cells of each row are stored next to it (see Index.h).
//
// BC_PERIODIC:     U[-1] = U[N],  U[N+1] = U[0]
// BC_TRANSMISSIVE: U[-1] = U[0],  U[N+1] = U[N]   (waves leave the domain: Sod, dam break)
typedef enum { BC_PERIODIC, BC_TRANSMISSIVE } BCType;

// Fills the ghost cells of the M rows of a state, once all its cells are known.
// U points to cell 0 of variable 0, and the rows are ROW_SIZE(N) apart.
// Used by the GPU on the first step and on the CPU on every step
inline HOST_DEVICE void fillGhosts(double* U, int N, int M, BCType bc)
{
    for(int var = 0; var < M; ++var) {
        int idx = (size_t)(var) * ROW_SIZE(N); // idx - 1 is the left ghost cell and idx + N + 1 the right ghost cell. U is pointing to the first non-ghost cell of the first variable
        switch (bc) {
        case BC_PERIODIC:
            U[idx - 1]     = U[idx + N]; // Left ghost is a copy of the last cell
            U[idx + N + 1] = U[idx];     // Right ghost is a copy of the first cell
            break;
        
        case BC_TRANSMISSIVE:
            U[idx - 1]     = U[idx];  // Waves escape the domain
            U[idx + N + 1] = U[idx + N];
            break;
        }
    }
}

// Writes the ghost cells of the M rows that depend on cell j, right after cell j has been computed.
// On GPU, it lets each GPU thread fill the ghosts of the state it is writing, without waiting for the other threads.
// Used only by GPU
inline HOST_DEVICE void fillGhostsFromCell(double* U, int j, int N, int M, BCType bc)
{
    for(int var = 0; var < M; ++var) {
        int idx = (size_t)(var) * ROW_SIZE(N); // idx - 1 is the left ghost cell and idx + N + 1 the right ghost cell. U is pointing to the first non-ghost cell of the first variable
        switch (bc) {
        case BC_PERIODIC:
            if (j == N) U[idx - 1]     = U[idx + N]; // Left ghost is a copy of the last cell
            if (j == 0) U[idx + N + 1] = U[idx];     // Right ghost is a copy of the first cell
            break;
        
        case BC_TRANSMISSIVE:
            if (j == 0) U[idx - 1]     = U[idx];  // Waves escape the domain
            if (j == N) U[idx + N + 1] = U[idx + N];
            break;
        }
    }
}

#endif
