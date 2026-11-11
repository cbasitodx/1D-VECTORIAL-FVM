#ifndef CFL_HPP
#define CFL_HPP

#include "include/Index.h"     // IDX, numRecords
#include "include/Physics.hpp" // PhysicsType

// The solvers use a fixed dt, chosen by the user from a speed bound. For systems, the maximum speed can grow
// during the run, so this function measures the CFL number that was really used: max(s)*dt/dx over the recorded
// states. It only sees recorded steps, so it is a check, not a guarantee. A value above 1 means the run is not valid.
template <PhysicsType Physics>
double effectiveCFL(
    const double* u,                       // IN: History array, as filled by the solver
    int N,                                 // IN: N+1 is the number of finite volumes
    int T,                                 // IN: Number of time steps of the simulation
    int recordEvery,                       // IN: Recorded every 'recordEvery' time steps
    const typename Physics::Params& params,// IN: Parameters of the equation
    double dt,                             // IN: Temporal time step
    double dx                              // IN: Spatial step
)
{
    // First, let's get the number of variables we are handling
    const int M = Physics::NVARS;

    // Let us obtain the number of records (blocks of M variables with N+1 cells each)
    const int nRecords = numRecords(T, recordEvery);

    // For each record and cell, gather the M variables of the cell and take the largest Physics::maxSpeed
    double maxWaveSpeed = 0.0; // Natural first estimative. Wave speed can't (as computed by Physics::maxWaveSpeed) can't be negative
    for(int r = 0; r < nRecords; ++r) {   // Iterate through the records
        for(int c_j = 0; c_j <= N; ++c_j) {  // Iterate through the cells
            double U[M];
            for(int k = 0; k < M; ++k)    { // Iterate through each variable
                U[k] = u[IDX(r,k,c_j,N,M)];
            }
            double s = Physics::maxSpeed(U, params);
            maxWaveSpeed = maxWaveSpeed > s? maxWaveSpeed : s; 
        }
    }

    return maxWaveSpeed * (dt/dx);
}

#endif
