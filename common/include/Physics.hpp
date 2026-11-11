#ifndef PHYSICS_HPP
#define PHYSICS_HPP

#if __cplusplus < 202002L
#error "This project needs C++20 (for the PhysicsType concept): compile with -std=c++20"
#endif

#include <concepts> // std::same_as, std::convertible_to (compile time only: no code is generated)

// =====================================================================================================================
// What is a "Physics" type?
// =====================================================================================================================
//
// The solvers (cpu/ and cuda/) solve any 1D system of conservation laws
//
//     U_t + f(U)_x = 0,      U = (U_0, ..., U_{M-1}): the M conserved variables
//
// but they know nothing about any particular equation. Everything that depends on the equation lives in a small
// struct, the "Physics" type, which is passed to the solver as a template parameter:
//
//     FVM1DVectorial<ShallowWater>(N, u0, params, xL, xR, dt, T, BC_PERIODIC, RECORD_EVERY, u);
//                    ^^^^^^^^^^^^ the Physics type
//
// Because it is a template parameter, the equation is fixed at compile time and its functions are inlined into the
// solver loop: there is no function call per cell, and no run-time cost for being generic.
//
// A Physics type is a struct with only static members (it is never instantiated), and it must provide four things:
//
//     struct MyEquation {
//         static constexpr int NVARS = ...;                                              // 1. M
//         struct Params { ... };                                                         // 2. Parameters
//         FORCE_INLINE static void   flux(const double* U, double* F, const Params& p);  // 3. f(U)
//         FORCE_INLINE static double maxSpeed(const double* U, const Params& p);         // 4. Largest |wave speed|
//     };
//
// 1. NVARS: the number of conserved variables M (1 for scalar equations).
//    It must be a compile-time constant (constexpr): the solver uses it to size small local arrays, double UL[M],
//    and the compiler unrolls the loops over the variables.
//
// 2. Params: the parameters of the equation that are only known at run time (gravity g, gamma, a velocity a, ...).
//    Use an empty struct if there are none: struct Params {};
//    It is passed by const reference on the CPU and copied by value to the GPU as a kernel argument, so it must
//    hold plain values only (no pointers to host memory).
//
// 3. flux(U, F, p): the physical flux of ONE cell. U points to the M VARIABLES of that cell, U[0..M-1], and
//    flux writes F[0..M-1] = f(U). All the components are computed at once because each one may need all the
//    variables (for shallow water, f_1 = hu^2/h + g h^2/2 needs both h and hu).
//
// 4. maxSpeed(U, p): the largest ABSOLUTE wave speed of ONE cell, max_k |lambda_k(U)|, where lambda_k are the
//    eigenvalues of the Jacobian f'(U). It must be >= 0: the direction of the waves does not matter, only how fast
//    they are. It is used by the Rusanov flux (as its numerical diffusion) and by effectiveCFL (CFL.hpp).
//    A closed formula is enough (|u| + sqrt(g h) for shallow water, |u| + c for Euler); no eigenvalue solver is
//    needed. An upper bound is also valid (more diffusion, still stable), but never a value below the true one
//    (the scheme can become unstable).
//
// =====================================================================================================================
// What the compiler checks, and what it cannot
// =====================================================================================================================
//
// The concept PhysicsType below checks the four points above at compile time, where the solver is called. If one
// is missing or has the wrong form, the compiler says which requirement is not satisfied. For example, a flux
// that takes double* U instead of const double* U gives (clang; gcc words it differently):
//
//     error: static assertion failed
//     note: because 'MyEquation' does not satisfy 'PhysicsType'
//     note: because 'P::flux(U, F, p)' would be invalid: cannot initialize a parameter of type 'double *'
//           with an lvalue of type 'const double *'
//
// The compiler CANNOT check these, so they are up to the user:
//   - FORCE_INLINE (common/include/Portability.h) on flux and maxSpeed. Without it, the code still works on the
//     CPU (maybe slower), but nvcc refuses to call the functions from a GPU kernel.
//   - That the formulas are right, and that maxSpeed returns an absolute value (>= 0).
//   - Physical validity: the solver does not check it. Shallow water needs h > 0, Euler rho > 0 and p > 0;
//     otherwise there are divisions by zero or square roots of negative numbers (NaN).
//   - GPU restrictions: flux and maxSpeed must not use global variables, host-only functions or printf.
//
// =====================================================================================================================
// Example: Burgers' equation, u_t + (u^2/2)_x = 0 (examples/burgers/Burgers.hpp)
// =====================================================================================================================
//
//     struct Burgers {
//         static constexpr int NVARS = 1;
//
//         struct Params {};                                    // No parameters
//
//         FORCE_INLINE static void flux(const double* U, double* F, const Params& p)
//         {
//             (void)p;                                         // Unused (avoids a compiler warning)
//             F[0] = 0.5*U[0]*U[0];                            // f(u) = u^2/2
//         }
//
//         FORCE_INLINE static double maxSpeed(const double* U, const Params& p)
//         {
//             (void)p;
//             return fabs(U[0]);                               // |f'(u)| = |u|
//         }
//     };
//     static_assert(PhysicsType<Burgers>);                     // Checks the struct right here, where it is written
//
// More examples: LinearTransport (one parameter), ShallowWater (M = 2) and Euler (M = 3), in examples/.
//
// =====================================================================================================================
// How to add a new example
// =====================================================================================================================
//
//   1. Create a folder examples/<name>/ with a header <Name>.hpp that defines the struct, as in the example above.
//      Include "include/Portability.h" (FORCE_INLINE) and "include/Physics.hpp" (PhysicsType).
//   2. Add static_assert(PhysicsType<Name>); after the struct, so mistakes are reported in your own file.
//   3. Write a test next to it (copy one of the tests in examples/) with an initial condition and a time step.
//      For systems, choose dt from a bound of maxSpeed over the whole run, and check the "Effective CFL" it prints
//      (it must stay below 1).
//   4. Add the folder name to EXAMPLES in examples/Makefile. That is the only change needed: every test*.cpp and
//      test*.cu inside a listed folder is then found and built automatically.

// The requirements of a Physics type, checked at compile time (see the explanation above)
template <class P>
concept PhysicsType =
    requires { typename P::Params; } &&                                // 2. Params
    requires { { P::NVARS } -> std::convertible_to<int>; } &&          // 1. NVARS (should be an int: a double would also pass) ...
    (P::NVARS >= 1) &&                                                 //    ... a compile-time constant, at least 1
    requires(const double* U, double* F, const typename P::Params& p) {
        { P::flux(U, F, p) }  -> std::same_as<void>;                   // 3. flux
        { P::maxSpeed(U, p) } -> std::same_as<double>;                 // 4. maxSpeed
    };

#endif
