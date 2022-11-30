#ifndef LBVH_AABB_CUH
#define LBVH_AABB_CUH
#include "utility.cuh"
#include <cmath>
#include <limits>
#include <thrust/swap.h>

namespace lbvh {

template <typename T, unsigned int dim> struct aabb {
    typename vector_of<T, dim>::type upper;
    typename vector_of<T, dim>::type lower;
};

template <typename T>
__device__ __host__ inline bool intersects(const aabb<T, 2> &lhs, const aabb<T, 2> &rhs) noexcept {
    if (lhs.upper.x < rhs.lower.x || rhs.upper.x < lhs.lower.x) { return false; }
    if (lhs.upper.y < rhs.lower.y || rhs.upper.y < lhs.lower.y) { return false; }
    return true;
}

template <typename T>
__device__ __host__ inline bool intersects(const aabb<T, 3> &lhs, const aabb<T, 3> &rhs) noexcept {
    if (lhs.upper.x < rhs.lower.x || rhs.upper.x < lhs.lower.x) { return false; }
    if (lhs.upper.y < rhs.lower.y || rhs.upper.y < lhs.lower.y) { return false; }
    if (lhs.upper.z < rhs.lower.z || rhs.upper.z < lhs.lower.z) { return false; }
    return true;
}

__device__ __host__ inline aabb<double, 2> merge(const aabb<double, 2> &lhs, const aabb<double, 2> &rhs) noexcept {
    aabb<double, 2> merged;
    merged.upper.x = ::fmax(lhs.upper.x, rhs.upper.x);
    merged.upper.y = ::fmax(lhs.upper.y, rhs.upper.y);
    merged.lower.x = ::fmin(lhs.lower.x, rhs.lower.x);
    merged.lower.y = ::fmin(lhs.lower.y, rhs.lower.y);
    return merged;
}

__device__ __host__ inline aabb<float, 2> merge(const aabb<float, 2> &lhs, const aabb<float, 2> &rhs) noexcept {
    aabb<float, 2> merged;
    merged.upper.x = ::fmax(lhs.upper.x, rhs.upper.x);
    merged.upper.y = ::fmax(lhs.upper.y, rhs.upper.y);
    merged.lower.x = ::fmin(lhs.lower.x, rhs.lower.x);
    merged.lower.y = ::fmin(lhs.lower.y, rhs.lower.y);
    return merged;
}

__device__ __host__ inline aabb<double, 3> merge(const aabb<double, 3> &lhs, const aabb<double, 3> &rhs) noexcept {
    aabb<double, 3> merged;
    merged.upper.x = ::fmax(lhs.upper.x, rhs.upper.x);
    merged.upper.y = ::fmax(lhs.upper.y, rhs.upper.y);
    merged.upper.z = ::fmax(lhs.upper.z, rhs.upper.z);
    merged.lower.x = ::fmin(lhs.lower.x, rhs.lower.x);
    merged.lower.y = ::fmin(lhs.lower.y, rhs.lower.y);
    merged.lower.z = ::fmin(lhs.lower.z, rhs.lower.z);
    return merged;
}

__device__ __host__ inline aabb<float, 3> merge(const aabb<float, 3> &lhs, const aabb<float, 3> &rhs) noexcept {
    aabb<float, 3> merged;
    merged.upper.x = ::fmax(lhs.upper.x, rhs.upper.x);
    merged.upper.y = ::fmax(lhs.upper.y, rhs.upper.y);
    merged.upper.z = ::fmax(lhs.upper.z, rhs.upper.z);
    merged.lower.x = ::fmin(lhs.lower.x, rhs.lower.x);
    merged.lower.y = ::fmin(lhs.lower.y, rhs.lower.y);
    merged.lower.z = ::fmin(lhs.lower.z, rhs.lower.z);
    return merged;
}

// metrics defined in
// Nearest Neighbor Queries (1995) ACS-SIGMOD
// - Nick Roussopoulos, Stephen Kelley FredericVincent

__device__ __host__ inline float mindist(const aabb<float, 2> &lhs, const float2 &rhs) noexcept {
    const float dx = ::fmin(lhs.upper.x, ::fmax(lhs.lower.x, rhs.x)) - rhs.x;
    const float dy = ::fmin(lhs.upper.y, ::fmax(lhs.lower.y, rhs.y)) - rhs.y;
    return dx * dx + dy * dy;
}

__device__ __host__ inline double mindist(const aabb<double, 2> &lhs, const double2 &rhs) noexcept {
    const double dx = ::fmin(lhs.upper.x, ::fmax(lhs.lower.x, rhs.x)) - rhs.x;
    const double dy = ::fmin(lhs.upper.y, ::fmax(lhs.lower.y, rhs.y)) - rhs.y;
    return dx * dx + dy * dy;
}

__device__ __host__ inline float mindist(const aabb<float, 3> &lhs, const float4 &rhs) noexcept {
    const float dx = ::fmin(lhs.upper.x, ::fmax(lhs.lower.x, rhs.x)) - rhs.x;
    const float dy = ::fmin(lhs.upper.y, ::fmax(lhs.lower.y, rhs.y)) - rhs.y;
    const float dz = ::fmin(lhs.upper.z, ::fmax(lhs.lower.z, rhs.z)) - rhs.z;
    return dx * dx + dy * dy + dz * dz;
}

__device__ __host__ inline double mindist(const aabb<double, 3> &lhs, const double4 &rhs) noexcept {
    const double dx = ::fmin(lhs.upper.x, ::fmax(lhs.lower.x, rhs.x)) - rhs.x;
    const double dy = ::fmin(lhs.upper.y, ::fmax(lhs.lower.y, rhs.y)) - rhs.y;
    const double dz = ::fmin(lhs.upper.z, ::fmax(lhs.lower.z, rhs.z)) - rhs.z;
    return dx * dx + dy * dy + dz * dz;
}

__device__ __host__ inline float minmaxdist(const aabb<float, 2> &lhs, const float2 &rhs) noexcept {
    float2 rm_sq =
        make_float2((lhs.lower.x - rhs.x) * (lhs.lower.x - rhs.x), (lhs.lower.y - rhs.y) * (lhs.lower.y - rhs.y));
    float2 rM_sq =
        make_float2((lhs.upper.x - rhs.x) * (lhs.upper.x - rhs.x), (lhs.upper.y - rhs.y) * (lhs.upper.y - rhs.y));

    if ((lhs.upper.x + lhs.lower.x) * 0.5f < rhs.x) { thrust::swap(rm_sq.x, rM_sq.x); }
    if ((lhs.upper.y + lhs.lower.y) * 0.5f < rhs.y) { thrust::swap(rm_sq.y, rM_sq.y); }

    const float dx = rm_sq.x + rM_sq.y;
    const float dy = rM_sq.x + rm_sq.y;

    return ::fmin(dx, dy);
}

__device__ __host__ inline float minmaxdist(const aabb<float, 3> &lhs, const float4 &rhs) noexcept {
    float3 rm_sq = make_float3(
        (lhs.lower.x - rhs.x) * (lhs.lower.x - rhs.x), (lhs.lower.y - rhs.y) * (lhs.lower.y - rhs.y),
        (lhs.lower.z - rhs.z) * (lhs.lower.z - rhs.z)
    );
    float3 rM_sq = make_float3(
        (lhs.upper.x - rhs.x) * (lhs.upper.x - rhs.x), (lhs.upper.y - rhs.y) * (lhs.upper.y - rhs.y),
        (lhs.upper.z - rhs.z) * (lhs.upper.z - rhs.z)
    );

    if ((lhs.upper.x + lhs.lower.x) * 0.5f < rhs.x) { thrust::swap(rm_sq.x, rM_sq.x); }
    if ((lhs.upper.y + lhs.lower.y) * 0.5f < rhs.y) { thrust::swap(rm_sq.y, rM_sq.y); }
    if ((lhs.upper.z + lhs.lower.z) * 0.5f < rhs.z) { thrust::swap(rm_sq.z, rM_sq.z); }

    const float dx = rm_sq.x + rM_sq.y + rM_sq.z;
    const float dy = rM_sq.x + rm_sq.y + rM_sq.z;
    const float dz = rM_sq.x + rM_sq.y + rm_sq.z;

    return ::fmin(dx, ::fmin(dy, dz));
}

__device__ __host__ inline double minmaxdist(const aabb<double, 2> &lhs, const double2 &rhs) noexcept {
    double2 rm_sq =
        make_double2((lhs.lower.x - rhs.x) * (lhs.lower.x - rhs.x), (lhs.lower.y - rhs.y) * (lhs.lower.y - rhs.y));
    double2 rM_sq =
        make_double2((lhs.upper.x - rhs.x) * (lhs.upper.x - rhs.x), (lhs.upper.y - rhs.y) * (lhs.upper.y - rhs.y));

    if ((lhs.upper.x + lhs.lower.x) * 0.5 < rhs.x) { thrust::swap(rm_sq.x, rM_sq.x); }
    if ((lhs.upper.y + lhs.lower.y) * 0.5 < rhs.y) { thrust::swap(rm_sq.y, rM_sq.y); }

    const double dx = rm_sq.x + rM_sq.y;
    const double dy = rM_sq.x + rm_sq.y;

    return ::fmin(dx, dy);
}

__device__ __host__ inline double minmaxdist(const aabb<double, 3> &lhs, const double4 &rhs) noexcept {
    double3 rm_sq = make_double3(
        (lhs.lower.x - rhs.x) * (lhs.lower.x - rhs.x), (lhs.lower.y - rhs.y) * (lhs.lower.y - rhs.y),
        (lhs.lower.z - rhs.z) * (lhs.lower.z - rhs.z)
    );
    double3 rM_sq = make_double3(
        (lhs.upper.x - rhs.x) * (lhs.upper.x - rhs.x), (lhs.upper.y - rhs.y) * (lhs.upper.y - rhs.y),
        (lhs.upper.z - rhs.z) * (lhs.upper.z - rhs.z)
    );

    if ((lhs.upper.x + lhs.lower.x) * 0.5 < rhs.x) { thrust::swap(rm_sq.x, rM_sq.x); }
    if ((lhs.upper.y + lhs.lower.y) * 0.5 < rhs.y) { thrust::swap(rm_sq.y, rM_sq.y); }
    if ((lhs.upper.z + lhs.lower.z) * 0.5 < rhs.z) { thrust::swap(rm_sq.z, rM_sq.z); }

    const double dx = rm_sq.x + rM_sq.y + rM_sq.z;
    const double dy = rM_sq.x + rm_sq.y + rM_sq.z;
    const double dz = rM_sq.x + rM_sq.y + rm_sq.z;

    return ::fmin(dx, ::fmin(dy, dz));
}

template <typename T>
__device__ __host__ inline typename vector_of<T, 2>::type centroid(const aabb<T, 2> &box) noexcept {
    typename vector_of<T, 2>::type c;
    c.x = (box.upper.x + box.lower.x) * 0.5;
    c.y = (box.upper.y + box.lower.y) * 0.5;
    return c;
}

template <typename T>
__device__ __host__ inline typename vector_of<T, 3>::type centroid(const aabb<T, 3> &box) noexcept {
    typename vector_of<T, 3>::type c;
    c.x = (box.upper.x + box.lower.x) * 0.5;
    c.y = (box.upper.y + box.lower.y) * 0.5;
    c.z = (box.upper.z + box.lower.z) * 0.5;
    return c;
}

template <typename T, unsigned int dim> struct Line {
    typename vector_of<T, dim>::type origin;
    typename vector_of<T, dim>::type dir;
};

// refeence: https://github.com/gszauer/GamePhysicsCookbook/blob/master/Code/Geometry3D.cpp
template <typename T>
__device__ __host__ inline bool intersects(const Line<T, 3> &line, const aabb<T, 3> &aabb) noexcept {
    constexpr T eps = std::numeric_limits<T>::epsilon();
    float t1 = (aabb.lower.x - line.origin.x) / ((-eps <= line.dir.x && line.dir.x <= eps) ? eps : line.dir.x);
    float t2 = (aabb.upper.x - line.origin.x) / ((-eps <= line.dir.x && line.dir.x <= eps) ? eps : line.dir.x);
    float t3 = (aabb.lower.y - line.origin.y) / ((-eps <= line.dir.y && line.dir.y <= eps) ? eps : line.dir.y);
    float t4 = (aabb.upper.y - line.origin.y) / ((-eps <= line.dir.y && line.dir.y <= eps) ? eps : line.dir.y);
    float t5 = (aabb.lower.z - line.origin.z) / ((-eps <= line.dir.z && line.dir.z <= eps) ? eps : line.dir.z);
    float t6 = (aabb.upper.z - line.origin.z) / ((-eps <= line.dir.z && line.dir.z <= eps) ? eps : line.dir.z);

    float tmin = fmax(fmax(fmin(t1, t2), fmin(t3, t4)), fmin(t5, t6));
    float tmax = fmin(fmin(fmax(t1, t2), fmax(t3, t4)), fmax(t5, t6));

    if (tmin > tmax) return false;
    return true;
}

template <typename T>
__device__ __host__ inline bool intersects(const Line<T, 2> &line, const aabb<T, 2> &aabb) noexcept {
    constexpr T eps = std::numeric_limits<T>::epsilon();
    float t1 = (aabb.lower.x - line.origin.x) / ((-eps <= line.dir.x && line.dir.x <= eps) ? eps : line.dir.x);
    float t2 = (aabb.upper.x - line.origin.x) / ((-eps <= line.dir.x && line.dir.x <= eps) ? eps : line.dir.x);
    float t3 = (aabb.lower.y - line.origin.y) / ((-eps <= line.dir.y && line.dir.y <= eps) ? eps : line.dir.y);
    float t4 = (aabb.upper.y - line.origin.y) / ((-eps <= line.dir.y && line.dir.y <= eps) ? eps : line.dir.y);

    float tmin = fmax(fmin(t1, t2), fmin(t3, t4));
    float tmax = fmin(fmax(t1, t2), fmax(t3, t4));

    if (tmin > tmax) return false;
    return true;
}

} // namespace lbvh
#endif // LBVH_AABB_CUH
