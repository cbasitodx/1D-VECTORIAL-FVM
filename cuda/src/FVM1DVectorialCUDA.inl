// Implementation of FVM1DVectorialCUDA. Not compiled on its own: included at the end of include/FVM1DVectorialCUDA.cuh

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

// Macro. Abort with a message if a CUDA API call fails
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err_ = (call);                                              \
        if (err_ != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,    \
                    cudaGetErrorString(err_));                                  \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

// One time step, fused. uo and un point to cell 0 of variable 0, and the rows are ROW_SIZE(N) apart.
// No thread depends on another one: all of them only read the previous state, which the previous launch completed.
template <PhysicsType Physics, class NumFlux>
__global__ void stepKernel(const double* __restrict__ uo, double* __restrict__ un, int N, double lambda, BCType bc,
                           typename Physics::Params params)
{
    const int    M   = Physics::NVARS;
    const ptrdiff_t row = ROW_SIZE(N);

    int globalThreadIdx = blockIdx.x * blockDim.x + threadIdx.x; // Index of the thread within the grid
    int numThreadsInGrid = gridDim.x * blockDim.x;               // Number of threads in the grid
    for(int j = globalThreadIdx; j <= N; j+=numThreadsInGrid) { // grid-stride loop
        // Gathering of the M variables of cells j-1, j and j+1 (the ghost cells are already filled by the previous kernel)
        double UL[M], UR[M], U[M];
        for(int k = 0; k < M; ++k) {
            UL[k] = uo[k*row + j-1];
            U [k] = uo[k*row + j];
            UR[k] = uo[k*row + j+1]; 
        }
        
        // Evaluation of the flux and maxSpeed of the three cells
        double FL[M], F[M], FR[M];
        Physics::flux(UL, FL, params);
        Physics::flux(U , F , params);
        Physics::flux(UR, FR, params);

        double sL, s, sR;
        sL = Physics::maxSpeed(UL, params);
        s  = Physics::maxSpeed(U, params);
        sR = Physics::maxSpeed(UR, params);

        double halfSpeedL = NumFlux::halfSpeed(sL, s);
        double halfSpeedR = NumFlux::halfSpeed(s, sR);

        // Computation of the numerical fluxes at interfaces j-1/2 and j+1/2 with NumFlux
        double numFL[M], numFR[M];
        for (int k = 0; k < M; ++k) { 
            numFL[k] = NumFlux::F(UL[k], U[k] , FL[k], F[k], halfSpeedL); // F_{j-1/2}
            numFR[k] = NumFlux::F(U[k] , UR[k], F[k], FR[k], halfSpeedR); // F_{j+1/2}
        }

        // State update
        for (int k = 0; k < M; ++k) { un[k*row + j] = uo[k*row + j] - lambda*(numFR[k] - numFL[k]); }

        // If it corresponds, update the ghost cells for the new state un
        fillGhostsFromCell(un, j, N, M, bc);
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// Shared-memory version of stepKernel: the physics (flux and maxSpeed) of each cell is evaluated once instead of
// three times. Each block works on a tile of blockDim.x consecutive cells, in two phases:
//   1. each thread evaluates the physics of its own cell and stores U, F and s in shared memory (the edge threads also
//      evaluate the two halo cells, the neighbours just outside the tile);
//   2. after a barrier, each thread reads its neighbours from shared memory, computes its two interfaces and updates
//      its cell.
//
// Note: A tile is just a group of consecutive cells a block works on at the same time
// ---------------------------------------------------------------------------------------------------------------------

// For one cell: reads the M variables of cell 'cell' from uo, evaluates its flux and max speed, and stores
// U, F and s in slot 'slot' of the shared arrays (see stepKernelShared for their layout)
template <PhysicsType Physics>
__device__ __forceinline__ void evaluateCell(const double* __restrict__ uo,          // IN: Old state (cell 0 of variable 0)
                                             ptrdiff_t row,                          // IN: ROW_SIZE(N)
                                             int cell,                               // IN: Cell to evaluate, index in uo (-1 and N+1 are the ghost cells)
                                             double* sU,                             // OUT: Shared states, M rows of 'width' slots
                                             double* sF,                             // OUT: Shared physical fluxes, M rows of 'width' slots
                                             double* sS,                             // OUT: Shared max wave speeds, one per slot
                                             int slot,                               // IN: Where to store the cell, index in the shared arrays (0 is the left halo)
                                             int width,                              // IN: Slots per variable in shared memory (the tile's cells and one halo cell on each side)
                                             const typename Physics::Params& params  // IN: Parameters of the equation
)
{
    const int M = Physics::NVARS;

    double U[M];
    for (int k = 0; k < M; ++k) { U[k] = uo[k*row + cell]; }

    double F[M], s;
    Physics::flux(U, F, params);
    s = Physics::maxSpeed(U, params);

    // Store computed cell values in shared memory: variable k of this slot is sU[k*width + slot] (same for sF), and s is sS[slot]
    for (int k = 0; k < M; ++k) {
        sU[k*width + slot] = U[k]; 
        sF[k*width + slot] = F[k];
    }
    sS[slot] = s;
}

// One time step, fused, with shared memory. Same arguments as stepKernel. Launch with a third parameter, the bytes of
// shared memory per block:
//     stepKernelShared<Physics, NumFlux><<<numBlocks, threadsPerBlock, sharedBytes>>>(...)
//     sharedBytes = (2*M + 1)*(threadsPerBlock + 2)*sizeof(double)
template <PhysicsType Physics, class NumFlux>
__global__ void stepKernelShared(const double* __restrict__ uo, double* __restrict__ un, int N, double lambda, BCType bc,
                                 typename Physics::Params params)
{
    const int       M     = Physics::NVARS;
    const ptrdiff_t row   = ROW_SIZE(N);
    const int       width = blockDim.x + 2; // Slots per variable in shared memory: the tile's cells and one halo (cells outside the tile, not necessarily ghosts) cell on each side

    // Shared memory of the block (its size is given at launch): one flat array, split in three parts like the rows of a state
    //   sU: M rows of 'width' slots (states)  |  sF: M rows of 'width' slots (fluxes)  |  sS: 1 row of 'width' slots (speeds)
    extern __shared__ double shared[];
    double* sU = shared;
    double* sF = sU + M*width;
    double* sS = sF + M*width;

    const int i = threadIdx.x + 1; // My slot. Slot 0 is the left halo and slot blockDim.x + 1 the right halo

    // Loop over the tiles of the domain. 'base' (the first cell of the tile) is the same for all the threads of the block,
    // so all of them make the same iterations and reach the two __syncthreads() below.
    // The last tile may be partly empty (the domain ends inside it): its threads past N+1 have no cell to evaluate
    // and past N no cell to update, but they still go through the loop and take part in the barriers
    for (int base = blockIdx.x*blockDim.x; base <= N; base += gridDim.x*blockDim.x) {
        const int j = base + threadIdx.x; // My cell. In the last tile it may be past N: then I only take part in the barriers

        // ---- Physics of the tile's cells, once each ----

        // If my cell exists (j <= N + 1; cell N+1 is the right ghost cell, needed as a halo), evaluate it into my slot i
        if(j <= N + 1) { evaluateCell<Physics>(uo, row, j, sU, sF, sS, i, width, params); }

        // Thread 0 (of this block) also evaluates the left halo: cell base - 1, into slot 0
        if(threadIdx.x == 0) { evaluateCell<Physics>(uo, row, base - 1, sU, sF, sS, 0, width, params); }

        // Right halo: the cell just after the tile (base + blockDim.x), stored in the last slot (blockDim.x + 1).
        // Only the last thread of the block does it, and only if that cell exists, i.e. it is at most N+1 (the right ghost cell).
        // It does not exist only in the last tile of the domain, when that tile is not full: then cell N+1 is inside the tile
        // and was already evaluated by its own thread in the first if
        if(threadIdx.x == blockDim.x - 1 && base + (int)blockDim.x <= N + 1) { 
            evaluateCell<Physics>(uo, row, base + blockDim.x, sU, sF, sS, blockDim.x + 1, width, params); 
        }

        __syncthreads(); // All the physics of the tile is in shared memory now

        // ---- My two interfaces (numerical fluxes) and my update, reading the neighbours from shared memory ----
        if (j <= N) {
            // The two half speeds, from sS[i-1], sS[i] and sS[i+1]
            double halfSpeedL = NumFlux::halfSpeed(sS[i-1], sS[i]);
            double halfSpeedR = NumFlux::halfSpeed(sS[i], sS[i+1]);

            // For each variable k, the numerical fluxes at j-1/2 (slots i-1 and i) and j+1/2 (slots i and i+1),
            // and the update of un[k*row + j] (the old value is in shared memory too)
            double numFL[M], numFR[M];
            for (int k = 0; k < M; ++k) { 
                numFL[k] = NumFlux::F(sU[k*width + i-1], sU[k*width + i]  , sF[k*width + i-1], sF[k*width + i]  , halfSpeedL); // F_{j-1/2}
                numFR[k] = NumFlux::F(sU[k*width + i]  , sU[k*width + i+1], sF[k*width + i]  , sF[k*width + i+1], halfSpeedR); // F_{j+1/2}
            }

            for(int k = 0; k < M; ++k) { un[k*row + j] = sU[k*width + i] - lambda*(numFR[k] - numFL[k]); }

            fillGhostsFromCell(un, j, N, M, bc);
        }

        __syncthreads(); // Nobody starts the next tile (overwriting the shared arrays) until everybody has finished this one
    }
}

template <PhysicsType Physics, class NumFlux>
void FVM1DVectorialCUDA(int N, const double* u_0, const typename Physics::Params& params, double xL, double xR, double dt,
                        int T, BCType bc, int recordEvery, double* u, int threadsPerBlock, int numBlocks, double* kernelSeconds)
{

    const int    M   = Physics::NVARS;

    double dx       = (xR - xL)/(N+1);
    double lambda   = dt/dx;

    // Shared memory per block for stepKernelShared: 2M+1 rows ([ sU: M rows ][ sF: M rows ][ sS: 1 row ]) of threadsPerBlock+2 slots (tile + 2 halo cells)
    size_t sharedBytes = (2*M + 1)*(threadsPerBlock + 2)*sizeof(double);

    // Device memory: The old state (t^n) and the new state (t^{n+1}).
    // uOld[k*row] and uNew[k*row] (k = 0, ..., M-1) point to cell 0 of their var array (row), 
    // so uOld[k*row - 1] and uOld[k*row + N+1] are its ghost cells (same for uNew)
    double* block;
    CUDA_CHECK(cudaMalloc((void**)&block, sizeof(double) * 2 * STATE_SIZE(N, M)));
    double* uOld = block + 1;
    double* uNew = block + 1 + STATE_SIZE(N,M);

    // First M rows with its ghost cells: built on the host, stored in the host history (it is always recorded)
    // and copied to the device as the first old state
    for (int k = 0; k < M; ++k) { memcpy(u + IDX(0, k, 0, N, M), u_0 + k*(N+1), sizeof(double)*(N+1)); }
    fillGhosts(u + IDX(0, 0, 0, N, M), N, M, bc); // pointer to cell 0 of variable 0
    CUDA_CHECK(cudaMemcpy(uOld - 1, u + IDX(0, 0, -1, N, M), sizeof(double)*STATE_SIZE(N, M), cudaMemcpyHostToDevice)); // -1: start at the first ghost cell  
          
    // Time loop on the host: one kernel launch per step. A launch starts only when the previous one has finished,
    // which is the only synchronization the scheme needs (each step reads the whole previous row).
    // Kernel time is measured in segments between copies, so the copies are not counted in it
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    double kernelMs = 0.0;
    CUDA_CHECK(cudaEventRecord(start));

    for (int t_n = 0; t_n < T - 1; ++t_n) {
        // Compute t^{n+1} from t^n

        // --- No shared memory version ---
        // stepKernel<Physics, NumFlux><<<numBlocks, threadsPerBlock>>>(uOld, uNew, N, lambda, bc, params);

        // --- Shared memory version ---
        stepKernelShared<Physics, NumFlux><<<numBlocks, threadsPerBlock, sharedBytes>>>(uOld, uNew, N, lambda, bc, params);

        if (isRecorded(t_n + 1, T, recordEvery)) { // t_n + 1 because first state is always recorded
            // Close the kernel time segment (this also waits for the kernels and reports their errors)
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaEventRecord(stop));
            CUDA_CHECK(cudaEventSynchronize(stop));
            float ms;
            CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
            kernelMs += ms;

            // Copy the new row (ghost cells included) to its place in the host history
            CUDA_CHECK(cudaMemcpy(u + IDX(recordRow(t_n + 1, recordEvery), 0, -1, N, M), uNew - 1, sizeof(double) * STATE_SIZE(N,M), cudaMemcpyDeviceToHost));

            // Open the next kernel time segment
            CUDA_CHECK(cudaEventRecord(start));
        }

        // Swap the rows: the new state becomes the old one for the next step,
        // and the old row (no longer needed) will be overwritten by the next new state
        double* tmp = uOld;
        uOld = uNew;
        uNew = tmp;
    }

    // Wait for the GPU to finish. It is already idle here (the last step is always recorded, so it was
    // synchronized and copied above); kept as a safeguard in case the recording rule changes
    CUDA_CHECK(cudaDeviceSynchronize());

    if (kernelSeconds != NULL) *kernelSeconds = 1e-3*kernelMs;
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(block));

}
