#ifndef FVM1D_VECTORIAL_HPP
#define FVM1D_VECTORIAL_HPP

#include "include/Index.h"              // IDX, HISTORY_SIZE, isRecorded, recordRow (common/)
#include "include/BoundaryConditions.h" // BCType, fillGhosts (common/)
#include "include/Rusanov.hpp"          // Rusanov (common/)
#include "include/Physics.hpp"          // PhysicsType (common/)

// CPU solver for 1D systems of conservation laws, single core: one pass over the cells per time step.
// The equation and the numerical flux are template parameters, so they are fixed at compile time and inlined
// into the solver loop. It works on two states (the old and the new one, which swap roles every step)
// and copies only the recorded time steps into the history.
//
// The template parameter Physics is any struct that satisfies PhysicsType (see common/include/Physics.hpp for
// what it is and how to write one, and examples/ for concrete equations):
//
//     static constexpr int NVARS;     Number of conserved variables (M)
//     struct Params;                  Parameters of the equation
//     FORCE_INLINE static void   flux(const double* U, double* F, const Params& p);   F = f(U), for one cell
//     FORCE_INLINE static double maxSpeed(const double* U, const Params& p);          max_k |lambda_k(U)|, for one cell
//
// Usage: Euler::Params params = { 1.4 };
//        FVM1DVectorial<Euler>(N, u0, params, xL, xR, dt, T, BC_TRANSMISSIVE, RECORD_EVERY, u);
template <PhysicsType Physics, class NumFlux = Rusanov>
void FVM1DVectorial(
    int N,                                  // IN: N+1 is the number of finite volumes, and N is the number of internal points between xL and xR
    const double* __restrict__ u_0,         // IN: Initial state: M blocks of N+1 elements, u_0[k*(N+1) + j]. Is the function averaged in the N+1 volumes.
                                            // __restrict__ guarantees no pointer aliasing between u_0 and u
    const typename Physics::Params& params, // IN: Parameters of the equation
    double xL,                              // IN: Left interval extrema  (xL < xR)
    double xR,                              // IN: Right interval extrema (xL < xR)
    double dt,                              // IN: Temporal time step (CFL is not verified by the solver, see CFL.hpp)
    int T,                                  // IN: Number of time steps, including the initial state: t^0, ..., t^{T-1}
                                            //     (initial time is 0, so the final time is dt*(T-1))
    BCType bc,                              // IN: Boundary condition type
    int recordEvery,                        // IN: Store only the time steps n with isRecorded(n, T, recordEvery) (see Index.h)
    double* __restrict__ u                  // OUT: History array of HISTORY_SIZE(numRecords(T, recordEvery), N, M) elements, allocated by the caller.
                                            //      Must not overlap u_0 (__restrict__). u[IDX(r, k, j, N, M)] is variable k at cell c_j and record r
);

// Implementation (in the header: template code must be visible where it is used)
#include "src/FVM1DVectorial.inl"

#endif
