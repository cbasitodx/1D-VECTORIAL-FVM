#ifndef PORTABILITY_H
#define PORTABILITY_H

// Lets the same code compile for the CPU (g++) and for the GPU (nvcc).
//
// HOST_DEVICE:  the function can be called on the CPU and inside GPU kernels. Expands to nothing on the CPU.
// FORCE_INLINE: same, and the function is always inlined. Used by everything called from the hot loops
//               (physics and numerical flux), where a real function call per cell would be too expensive.
#ifdef __CUDACC__
#define HOST_DEVICE  __host__ __device__
#define FORCE_INLINE __host__ __device__ __forceinline__
#else
#define HOST_DEVICE
#define FORCE_INLINE inline __attribute__((always_inline))
#endif

#endif
