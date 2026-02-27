#include "tensorflow/core/framework/bfloat16.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "utils_op.h"

namespace tensorflow {
namespace musa {

template <typename T>
void MusaSoftplusKernelLauncher(const void* in, void* out, int size,
                                musaStream_t stream);

template <typename T>
class MusaSoftplusOp : public MusaOpKernel {
 public:
  explicit MusaSoftplusOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& input = ctx->input(0);

    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, input.shape(), &output));

    if (input.NumElements() == 0) {
      return;
    }

    auto& handle = GetHandleByCtx(ctx);
    musaStream_t stream = (musaStream_t)handle.GetStream();
    MusaSoftplusKernelLauncher<T>(
        input.tensor_data().data(),
        const_cast<char*>(output->tensor_data().data()), input.NumElements(),
        stream);
  }
};

#define REGISTER_MUSA_SOFTPLUS(TYPE)                             \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("Softplus").Device("MUSA").TypeConstraint<TYPE>("T"), \
      MusaSoftplusOp<TYPE>);

REGISTER_MUSA_SOFTPLUS(float);
REGISTER_MUSA_SOFTPLUS(double);
REGISTER_MUSA_SOFTPLUS(Eigen::half);
REGISTER_MUSA_SOFTPLUS(bfloat16);

#undef REGISTER_MUSA_SOFTPLUS

}  // namespace musa
}  // namespace tensorflow
