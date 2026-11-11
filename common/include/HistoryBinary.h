#ifndef HISTORY_BINARY_H
#define HISTORY_BINARY_H

// Writes the history array u (as filled by the solvers) to a binary file. Ghost cells are not written.
//
// File layout (native byte order, little-endian on x86 and Apple Silicon):
//     int64   number of cells (N+1)
//     int64   number of variables (M)
//     int64   number of records (time steps written)
//     double  xL, xR            (cell j is centered at xL + (j + 0.5)*(xR - xL)/(N+1))
//     then, for each record:    double t, followed by M blocks of N+1 cell values (variable 0 first)
//
// Returns 0 on success, 1 if the file could not be opened or written.
int writeHistoryBinary(
    const char* filename, // IN: Output file name
    const double* u,      // IN: History array, u[IDX(r, k, j, N, M)]
    int N,                // IN: N+1 is the number of finite volumes
    int M,                // IN: Number of conserved variables
    int T,                // IN: Number of time steps of the simulation
    double xL,            // IN: Left interval extrema
    double xR,            // IN: Right interval extrema
    double dt,            // IN: Temporal time step
    int every             // IN: Recorded every 'every' time steps (see isRecorded in Index.h)
);

#endif
