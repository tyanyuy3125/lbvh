#ifndef LBVH_UTILITY_CUH
#define LBVH_UTILITY_CUH
#include <math_constants.h>
#include <vector_types.h>

namespace lbvh {

template <typename T, unsigned int dim> struct vector_of;
template <> struct vector_of<float, 2> {
    using type = float2;
};
template <> struct vector_of<double, 2> {
    using type = double2;
};
template <> struct vector_of<float, 3> {
    using type = float4;
};
template <> struct vector_of<double, 3> {
    using type = double4;
};

template <typename T, unsigned int dim> using vector_of_t = typename vector_of<T, dim>::type;

template <typename T> __device__ inline T infinity() noexcept;

template <> __device__ inline float infinity<float>() noexcept { return CUDART_INF_F; }
template <> __device__ inline double infinity<double>() noexcept { return CUDART_INF; }

} // namespace lbvh
#endif // LBVH_UTILITY_CUH
