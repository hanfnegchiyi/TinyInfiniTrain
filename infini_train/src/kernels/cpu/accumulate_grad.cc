#include <cstddef>
#include <memory>

#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::kernels::cpu {
void AccumulateGrad(const std::shared_ptr<Tensor> &gradient, float rate, const std::shared_ptr<Tensor> &tensor) {
    for (int64_t idx = 0; idx < gradient->NumElements(); ++idx) {
        static_cast<float *>(tensor->DataPtr())[idx] += rate * static_cast<const float *>(gradient->DataPtr())[idx];
    }
}

void AdamAccumulateGrad(const std::shared_ptr<Tensor> &grad, const std::shared_ptr<Tensor> &param,
                        const std::shared_ptr<Tensor> &m, const std::shared_ptr<Tensor> &v, float learning_rate,
                        float beta1, float beta2, float eps, int64_t t) {
    // =================================== 作业 ===================================
    // TODO：实现Adam优化器的梯度累积和参数更新
    // REF:
    // =================================== 作业 ===================================
    // 获取元素个数
    size_t n = param->NumElements();
    
    // 使用 DataPtr() 而不是 Data()
    float* grad_data = static_cast<float*>(grad->DataPtr());
    float* param_data = static_cast<float*>(param->DataPtr());
    float* m_data = static_cast<float*>(m->DataPtr());
    float* v_data = static_cast<float*>(v->DataPtr());
    
    // 预计算偏差校正
    float beta1_pow_t = std::pow(beta1, t);
    float beta2_pow_t = std::pow(beta2, t);
    float inv_bias_correction1 = 1.0f / (1.0f - beta1_pow_t);
    float inv_bias_correction2 = 1.0f / (1.0f - beta2_pow_t);
    
    // 循环更新每个元素
    for (size_t i = 0; i < n; ++i) {
        float gi = grad_data[i];
        
        // 更新一阶矩（动量）
        m_data[i] = beta1 * m_data[i] + (1.0f - beta1) * gi;
        
        // 更新二阶矩
        v_data[i] = beta2 * v_data[i] + (1.0f - beta2) * gi * gi;
        
        // 偏差校正后的估计
        float m_hat = m_data[i] * inv_bias_correction1;
        float v_hat = v_data[i] * inv_bias_correction2;
        
        // 更新参数
        param_data[i] -= learning_rate * m_hat / (std::sqrt(v_hat) + eps);
    }
}

} // namespace infini_train::kernels::cpu

#define REGISTER_CPU_ACCUMULATE_GRAD_KERNEL(kernel_name)                                                               \
    REGISTER_KERNEL(infini_train::DeviceType::kCPU, kernel_name, infini_train::kernels::cpu::kernel_name)

REGISTER_CPU_ACCUMULATE_GRAD_KERNEL(AccumulateGrad)
REGISTER_CPU_ACCUMULATE_GRAD_KERNEL(AdamAccumulateGrad)

#undef REGISTER_CPU_ACCUMULATE_GRAD_KERNEL
