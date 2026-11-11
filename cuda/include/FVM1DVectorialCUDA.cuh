#ifndef FVM1D_VECTORIAL_CUDA_CUH
#define FVM1D_VECTORIAL_CUDA_CUH

#include "include/Index.h"              // IDX, HISTORY_SIZE, isRecorded, recordRow (common/), same history layout as the CPU version
#include "include/BoundaryConditions.h" // BCType, fillGhosts, fillGhostsFromCell (common/)
#include "include/Rusanov.hpp"          // Rusanov (common/)
#include "include/Physics.hpp"          // PhysicsType (common/)

// CUDA version of FVM1DVectorial. Same arguments, history layout and template parameters as the CPU version
// (see common/include/Physics.hpp for what Physics must provide), except:
//   - only the recorded time steps are copied back to the host (see recordEvery),
//   - the launch configuration is given explicitly,
//   - the time spent in the kernels is returned, measured on the GPU.
//
// One fused kernel per time step. The GPU only holds two states (the old and the new one, which swap roles
// every step), so its memory does not limit T; each recorded state is copied to its place in the host history
// as soon as it is computed.
//
// Usage: Euler::Params params = { 1.4 };
//        FVM1DVectorialCUDA<Euler>(N, u0, params, xL, xR, dt, T, BC_TRANSMISSIVE, RECORD_EVERY, u,
//                                  threadsPerBlock, numBlocks, &kernelSeconds);
template <PhysicsType Physics, class NumFlux = Rusanov>
void FVM1DVectorialCUDA(
    int N,                                  // IN: N+1 is the number of finite volumes
    const double* u_0,                      // IN: Initial state: M blocks of N+1 elements, u_0[k*(N+1) + j] (host array)
    const typename Physics::Params& params, // IN: Parameters of the equation
    double xL,                              // IN: Left interval extrema  (xL < xR)
    double xR,                              // IN: Right interval extrema (xL < xR)
    double dt,                              // IN: Temporal time step (CFL is not verified by the solver, see CFL.hpp)
    int T,                                  // IN: Number of time steps, including the initial state
    BCType bc,                              // IN: Boundary condition type
    int recordEvery,                        // IN: Copy back only the time steps n with isRecorded(n, T, recordEvery) (see Index.h)
    double* u,                              // OUT: History array of HISTORY_SIZE(numRecords(T, recordEvery), N, M) elements (host array),
                                            //      u[IDX(r, k, j, N, M)] is variable k at cell c_j and record r
    int threadsPerBlock,                    // IN: Threads per block (multiple of 32)
    int numBlocks,                          // IN: Number of blocks (any value works: the kernel uses a grid-stride loop)
    double* kernelSeconds                   // OUT: Time spent in the kernels, measured on the GPU (copies excluded). May be NULL
);

// Implementation (in the header: template code must be visible where it is used)
#include "src/FVM1DVectorialCUDA.inl"

#endif
