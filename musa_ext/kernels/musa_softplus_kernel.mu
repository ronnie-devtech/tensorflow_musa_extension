#include <musa_fp16.h>
#include <musa_runtime.h>

#include <cmath>
#include <cstdint>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-pragmas"

#include "tensorflow/core/framework/bfloat16.h"
#include "tensorflow/core/framework/types.h"

#pragma GCC diagnostic pop

using bfloat16 = tensorflow::bfloat16;

namespace tensorflow {
namespace musa {

__device__ __forceinline__ float LoadFloat(const float* p) { return *p; }
__device__ __forceinline__ void StoreFloat(float* p, float v) { *p = v; }

__device__ __forceinline__ float LoadFloat(const Eigen::half* p) {
  const __half* h_ptr = reinterpret_cast<const __half*>(p);
  return __half2float(*h_ptr);
}

__device__ __forceinline__ void StoreFloat(Eigen::half* p, float v) {
  __half h = __float2half(v);
  *reinterpret_cast<__half*>(p) = h;
}

__device__ __forceinline__ float LoadFloat(const bfloat16* p) {
  float out = 0.0f;
  const uint16_t* b_ptr = reinterpret_cast<const uint16_t*>(p);
  uint32_t* f_ptr = reinterpret_cast<uint32_t*>(&out);
  *f_ptr = static_cast<uint32_t>(*b_ptr) << 16;
  return out;
}

__device__ __forceinline__ void StoreFloat(bfloat16* p, float v) {
  uint32_t* f_ptr = reinterpret_cast<uint32_t*>(&v);
  uint16_t b_val = static_cast<uint16_t>((*f_ptr) >> 16);
  *reinterpret_cast<uint16_t*>(p) = b_val;
}

__device__ __forceinline__ float SoftplusStable(float x) {
  return x > 0.0f ? (x + log1pf(expf(-x))) : log1pf(expf(x));
}

__device__ __forceinline__ double SoftplusStable(double x) {
  return x > 0.0 ? (x + log1p(exp(-x))) : log1p(exp(x));
}

template <typename T>
__global__ void SoftplusKernel(const T* in, T* out, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    float x = LoadFloat(in + idx);
    float y = SoftplusStable(x);
    StoreFloat(out + idx, y);
  }
}

template <>
__global__ void SoftplusKernel<double>(const double* in, double* out, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    out[idx] = SoftplusStable(in[idx]);
  }
}

template <typename T>
void MusaSoftplusKernelLauncher(const void* in, void* out, int size,
                                musaStream_t stream) {
  const int block_size = 256;
  const int grid_size = (size + block_size - 1) / block_size;
  SoftplusKernel<T><<<grid_size, block_size, 0, stream>>>(
      static_cast<const T*>(in), static_cast<T*>(out), size);
}

template void MusaSoftplusKernelLauncher<float>(const void*, void*, int,
                                                musaStream_t);
template void MusaSoftplusKernelLauncher<double>(const void*, void*, int,
                                                 musaStream_t);
template void MusaSoftplusKernelLauncher<Eigen::half>(const void*, void*, int,
                                                      musaStream_t);
template void MusaSoftplusKernelLauncher<bfloat16>(const void*, void*, int,
                                                   musaStream_t);

}  // namespace musa
}  // namespace tensorflow
