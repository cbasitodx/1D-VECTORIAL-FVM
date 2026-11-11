#ifndef INDEX_H
#define INDEX_H

#include <stddef.h>

// Layout of the flat history array u.
//
// A state is M consecutive rows, one per conserved variable. Each row has N+3 values (N + 1 is the number of volumens):
//
//     [ ghost | var 0: cell 0 ... cell N | ghost ] [ ghost | var 1: cell 0 ... cell N | ghost ] ...
//
// The ghost cells are stored in the row, so the solver can fill them in place (no copy of the row).
// The history only stores the recorded time steps, one state after the other (see recordRow).
//
// Usage: u[IDX(r, k, j, N, M)] is the cell average of variable k in cell c_j at record r, for j = 0..N.
//        j = -1 and j = N+1 are the ghost cells.

/*
Figure: IDX logic
position:   0   1   2   3   4 |  5   6   7   8   9 | 10  11  12  13  14 | 15  16  17  18  19
content:    g  c0  c1  c2   g |  g  c0  c1  c2   g |  g  c0  c1  c2   g |  g  c0  c1  c2   g
            └── var 0 ──────┘ └──── var 1 ───────┘ └──── var 0 ───────┘ └──── var 1 ───────┘
            └───────────── record 0 ─────────────┘ └───────────── record 1 ────────────────┘

 */
#define ROW_SIZE(N)           ((size_t)(N) + 3)
#define STATE_SIZE(N, M)      ((size_t)(M) * ROW_SIZE(N))
#define IDX(r, k, j, N, M)    ((size_t)(r) * STATE_SIZE(N, M) + (size_t)(k) * ROW_SIZE(N) + (size_t)((j) + 1)) // Notice that we need to index by the record because we are not keeping every single timestep

// Number of doubles to allocate for a history of R records (see numRecords), ghost cells included
#define HISTORY_SIZE(R, N, M) ((size_t)(R) * STATE_SIZE(N, M))

// True if time step n (of a history of T steps) is recorded when recording every 'every' steps.
// The first and last steps are always recorded. This is the single rule shared by the solvers and the binary writer.
inline int isRecorded(int n, int T, int every) { return (n % every == 0) || (n == T - 1); } // (n == T - 1 is the rule for the last step)

// Number of recorded time steps among the T steps
inline int numRecords(int T, int every) { return ((T - 1) + (every-1))/every + 1; } // (T - 1) + (every-1))/every counts the group of 'every' steps (rounding up unless exact multiple). The + 1 for the first step

// Position in the history of the recorded time step n. Only valid if isRecorded(n, T, every) == true
inline int recordRow(int n, int every) { return (n + every - 1)/every; } // Rounds up unless exact multiple

#endif
