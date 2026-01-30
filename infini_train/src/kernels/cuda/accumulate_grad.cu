#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::kernels::cuda {

__global__ void AccumulateGradKernel(const float *grad_ptr, float rate, float *tensor_ptr, size_t num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        tensor_ptr[idx] += rate * grad_ptr[idx];
    }
}

void AccumulateGrad(const std::shared_ptr<Tensor> &gradient, float rate, const std::shared_ptr<Tensor> &tensor) {
    size_t num_elements = gradient->NumElements();

    const float *grad_ptr = static_cast<const float *>(gradient->DataPtr());
    float *tensor_ptr = static_cast<float *>(tensor->DataPtr());

    int threads_per_block = 256;
    int num_blocks = (num_elements + threads_per_block - 1) / threads_per_block;

    AccumulateGradKernel<<<num_blocks, threads_per_block>>>(grad_ptr, rate, tensor_ptr, num_elements);
}

//cuda核函数
__global__ void adam_update_kernel(
    const float* __restrict__ grad,
    float* __restrict__ param,
    float* __restrict__ m,
    float* __restrict__ v,
    size_t n,
    float learning_rate,
    float beta1,
    float beta2,
    float eps,
    float inv_bias_correction1,
    float inv_bias_correction2) {
    
    // 计算线程索引
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 如果索引有效，处理对应的元素
    if (idx < n) {
        float g = grad[idx];
        
        // 更新一阶矩
        float m_val = m[idx];
        m_val = beta1 * m_val + (1.0f - beta1) * g;
        m[idx] = m_val;
        
        // 更新二阶矩
        float v_val = v[idx];
        v_val = beta2 * v_val + (1.0f - beta2) * g * g;
        v[idx] = v_val;
        
        // 计算偏差校正后的估计
        float m_hat = m_val * inv_bias_correction1;
        float v_hat = v_val * inv_bias_correction2;
        
        // 更新参数
        param[idx] -= learning_rate * m_hat / (sqrtf(v_hat) + eps);
    }
}

void AdamAccumulateGrad(const std::shared_ptr<Tensor> &grad, const std::shared_ptr<Tensor> &param,
                        const std::shared_ptr<Tensor> &m, const std::shared_ptr<Tensor> &v, float learning_rate,
                        float beta1, float beta2, float eps, int64_t t) {
    // =================================== 作业 ===================================
    // TODO：实现Adam优化器的梯度累积和参数更新
    // REF:
    // =================================== 作业 ===================================
    // 获取CUDA设备信息
    auto device = param->GetDevice();
    if (device.Type() != DeviceType::kCUDA) {
        throw std::runtime_error("This Adam CUDA kernel requires CUDA device");
    }
    
    // 获取数据指针
    float* grad_data = static_cast<float*>(grad->DataPtr());
    float* param_data = static_cast<float*>(param->DataPtr());
    float* m_data = static_cast<float*>(m->DataPtr());
    float* v_data = static_cast<float*>(v->DataPtr());
    
    // 获取元素数量
    size_t n = param->NumElements();
    
    // 设置线程块和网格大小
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    
    // 预计算偏差校正系数
    float beta1_pow_t = powf(beta1, static_cast<float>(t));
    float beta2_pow_t = powf(beta2, static_cast<float>(t));
    float inv_bias_correction1 = 1.0f / (1.0f - beta1_pow_t);
    float inv_bias_correction2 = 1.0f / (1.0f - beta2_pow_t);
    
    // 启动CUDA kernel
    adam_update_kernel<<<gridSize, blockSize>>>(
        grad_data, param_data, m_data, v_data,
        n, learning_rate, beta1, beta2, eps,
        inv_bias_correction1, inv_bias_correction2
    );
    
    // 检查CUDA错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA kernel failed: " + std::string(cudaGetErrorString(err)));
    }
}
} // namespace infini_train::kernels::cuda

#define REGISTER_CUDA_ACCUMULATE_GRAD_KERNEL(kernel_name)                                                              \
    REGISTER_KERNEL(infini_train::DeviceType::kCUDA, kernel_name, infini_train::kernels::cuda::kernel_name)

REGISTER_CUDA_ACCUMULATE_GRAD_KERNEL(AccumulateGrad)
REGISTER_CUDA_ACCUMULATE_GRAD_KERNEL(AdamAccumulateGrad)

#undef REGISTER_CUDA_ACCUMULATE_GRAD_KERNEL
