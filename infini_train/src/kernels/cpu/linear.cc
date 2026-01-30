#include <cstdint>
#include <fcntl.h>
#include <memory>
#include <numeric>
#include <tuple>

#include "glog/logging.h" 

#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::kernels::cpu {
std::shared_ptr<Tensor> MatmulForward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &other) {
    // =================================== 作业 ===================================
    // TODO：实现CPU上的矩阵乘法前向计算
    // REF:
    // =================================== 作业 ===================================
    const auto &input_dims = input->Dims();
    const auto &other_dims = other->Dims();
    
    // 基础维度校验
    CHECK_GE(input_dims.size(), 2);
    CHECK_GE(other_dims.size(), 2);
    
    // 提取矩阵维度
    const int64_t M = input_dims[input_dims.size() - 2];
    const int64_t K = input_dims[input_dims.size() - 1];
    const int64_t N = other_dims[other_dims.size() - 1];
    CHECK_EQ(K, other_dims[other_dims.size() - 2]);  // 维度匹配检查

    // 计算Batch Size
    int64_t batch_size = 1;
    for (size_t i = 0; i < input_dims.size() - 2; ++i) {
        batch_size *= input_dims[i];
    }
    
    // 检查两个输入的batch维度是否一致
    int64_t other_batch_size = 1;
    for (size_t i = 0; i < other_dims.size() - 2; ++i) {
        other_batch_size *= other_dims[i];
    }
    CHECK_EQ(batch_size, other_batch_size) << "MatmulForward requires same batch size for both inputs";

    // 创建输出Tensor
    auto output_dims = input_dims;
    output_dims[output_dims.size() - 1] = N;
    auto output = std::make_shared<Tensor>(output_dims, DataType::kFLOAT32);

    // 获取数据指针
    float *a = (float *)input->DataPtr();
    float *b = (float *)other->DataPtr();
    float *c = (float *)output->DataPtr();

    // 计算每个矩阵的大小
    const int64_t A_stride = M * K;
    const int64_t B_stride = K * N;
    const int64_t C_stride = M * N;

    // 按照每个batch计算矩阵乘法
    for (int64_t b_idx = 0; b_idx < batch_size; ++b_idx) {
        // 计算当前batch的指针偏移
        float *a_ptr = a + b_idx * A_stride;
        float *b_ptr = b + b_idx * B_stride;
        float *c_ptr = c + b_idx * C_stride;

        // 使用Eigen进行矩阵乘法计算
        Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> 
            Am(a_ptr, M, K);
        Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> 
            Bm(b_ptr, K, N);
        Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> 
            Cm(c_ptr, M, N);

        // 执行矩阵乘法：C = A * B
        Cm.noalias() = Am * Bm;
    }

    return output;
}

std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>
MatmulBackward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &other,
               const std::shared_ptr<Tensor> &grad_output) {
    // =================================== 作业 ===================================
    // TODO：实现CPU上的矩阵乘法反向传播
    // REF:
    // =================================== 作业 ===================================
    // 维度检查
    const auto &in_dims = input->Dims();
    const auto &ot_dims = other->Dims();
    const auto &go_dims = grad_output->Dims();
    
    CHECK_GE(in_dims.size(), 2);
    CHECK_GE(ot_dims.size(), 2);
    CHECK_GE(go_dims.size(), 2);

    // 提取矩阵维度
    const int64_t M = in_dims[in_dims.size() - 2];
    const int64_t K = in_dims[in_dims.size() - 1];
    const int64_t N = ot_dims[ot_dims.size() - 1];
    
    // 维度匹配检查
    CHECK_EQ(K, ot_dims[ot_dims.size() - 2]);
    CHECK_EQ(go_dims[go_dims.size() - 2], M);
    CHECK_EQ(go_dims[go_dims.size() - 1], N);

    // 计算Batch Size
    int64_t batch_size = 1;
    for (size_t i = 0; i < in_dims.size() - 2; ++i) {
        batch_size *= in_dims[i];
    }
    
    // 检查所有输入的batch维度是否一致
    int64_t other_batch_size = 1;
    for (size_t i = 0; i < ot_dims.size() - 2; ++i) {
        other_batch_size *= ot_dims[i];
    }
    int64_t grad_batch_size = 1;
    for (size_t i = 0; i < go_dims.size() - 2; ++i) {
        grad_batch_size *= go_dims[i];
    }
    
    CHECK_EQ(batch_size, other_batch_size) << "MatmulBackward requires same batch size for input and other";
    CHECK_EQ(batch_size, grad_batch_size) << "MatmulBackward requires same batch size for grad_output";

    // 创建梯度张量
    auto grad_input = std::make_shared<Tensor>(in_dims, DataType::kFLOAT32);
    auto grad_other = std::make_shared<Tensor>(ot_dims, DataType::kFLOAT32);

    // 获取数据指针
    float *a  = (float *)input->DataPtr();
    float *b  = (float *)other->DataPtr();
    float *go = (float *)grad_output->DataPtr();
    float *gi = (float *)grad_input->DataPtr();
    float *gb = (float *)grad_other->DataPtr();

    // 计算每个矩阵的大小
    const int64_t A_stride = M * K;
    const int64_t B_stride = K * N;
    const int64_t C_stride = M * N;

    // 初始化梯度为0
    std::memset(gi, 0, sizeof(float) * batch_size * A_stride);
    std::memset(gb, 0, sizeof(float) * batch_size * B_stride);

    // 逐batch计算梯度
    for (int64_t b_idx = 0; b_idx < batch_size; ++b_idx) {
        // 计算当前batch的指针偏移
        float *a_ptr  = a  + b_idx * A_stride;
        float *b_ptr  = b  + b_idx * B_stride;
        float *go_ptr = go + b_idx * C_stride;
        float *gi_ptr = gi + b_idx * A_stride;
        float *gb_ptr = gb + b_idx * B_stride;

        // 将数据映射为Eigen矩阵
        Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> 
            Am(a_ptr, M, K);
        Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> 
            Bm(b_ptr, K, N);
        Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> 
            GOm(go_ptr, M, N);
        Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> 
            GIm(gi_ptr, M, K);
        Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> 
            GBm(gb_ptr, K, N);
        // grad_input = grad_output * other^T
        // grad_other = input^T * grad_output
        GIm.noalias() += GOm * Bm.transpose();
        GBm.noalias() += Am.transpose() * GOm;
    }

    return {grad_input, grad_other};
}

std::shared_ptr<Tensor> LinearForward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &weight,
                                      bool transpose, const std::shared_ptr<Tensor> &bias) {
    /*
    transpose:  output = input * weight^T + bias
    output[*, out_features] = input[*, in_features] * weight[out_features, in_features]^T + bias[out_features]

    !transpose: output = input * weight + bias
    output[*, out_features] = input[*, in_features] * weight[in_features, out_features] + bias[out_features]
    */

    const auto &input_dims = input->Dims();
    CHECK_GE(input_dims.size(), 2);
    const int64_t bs = std::accumulate(input_dims.rbegin() + 1, input_dims.rend(), 1, std::multiplies<int64_t>{});
    const int64_t in_features = *input_dims.rbegin();

    const auto &weight_dims = weight->Dims();
    CHECK_EQ(weight_dims.size(), 2);
    CHECK_EQ(in_features, weight_dims[transpose ? 1 : 0]);
    const int out_features = weight_dims[transpose ? 0 : 1];

    if (bias) {
        const auto &bias_dims = bias->Dims();
        CHECK_EQ(bias_dims.size(), 1);
        CHECK_EQ(bias_dims[0], out_features);
    }

    auto output_dims = input_dims;
    *output_dims.rbegin() = out_features;
    auto output = std::make_shared<Tensor>(output_dims, DataType::kFLOAT32);

    if (transpose) {
        output->EigenMatrix() = input->EigenMatrix() * weight->EigenMatrix().transpose();
    } else {
        output->EigenMatrix() = input->EigenMatrix() * weight->EigenMatrix();
    }

    if (bias) {
        output->EigenMatrix().rowwise() += bias->EigenVector();
    }

    return output;
}

std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>
LinearBackward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &weight, bool transpose,
               int64_t out_features, const std::shared_ptr<Tensor> &grad_output, const bool bias) {
    /*
    transpose: grad_input = grad_output * weight
    grad_input[*, in_features] = grad_output[*, out_features] * weight[out_features, in_features]
    grad_weight[out_features, in_features] = grad_output[*, out_features]^T * input[*, in_features]
    grad_bias[out_features] = grad_output[*, out_features].sum(axis=0)

    !transpose: grad_input = grad_output * weight^T
    grad_input[*, in_features] = grad_output[_, out_features] * weight[in_features, out_features]^T
    grad_weight[in_features, out_features] = input[*, in_features]^T * grad_output[*, out_features]
    grad_bias[out_features] = grad_output[*, out_features].sum(axis=0)
    */

    const auto &input_dims = input->Dims();
    CHECK_GE(input_dims.size(), 2);
    const int64_t bs = std::accumulate(input_dims.rbegin() + 1, input_dims.rend(), 1, std::multiplies<int64_t>{});
    const int64_t in_features = *input_dims.rbegin();

    const auto &weight_dims = weight->Dims();
    CHECK_EQ(weight_dims.size(), 2);
    CHECK_EQ(in_features, weight_dims[transpose ? 1 : 0]);
    CHECK_EQ(out_features, weight_dims[transpose ? 0 : 1]);

    auto grad_input = std::make_shared<Tensor>(input_dims, DataType::kFLOAT32);
    auto grad_weight = std::make_shared<Tensor>(weight_dims, DataType::kFLOAT32);
    std::shared_ptr<Tensor> grad_bias = nullptr;
    if (bias) {
        grad_bias = std::make_shared<Tensor>(std::vector<int64_t>{out_features}, DataType::kFLOAT32);
    }

    if (transpose) {
        grad_input->EigenMatrix() = grad_output->EigenMatrix() * weight->EigenMatrix();
        grad_weight->EigenMatrix() = grad_output->EigenMatrix().transpose() * input->EigenMatrix();
    } else {
        grad_input->EigenMatrix() = grad_output->EigenMatrix() * weight->EigenMatrix().transpose();
        grad_weight->EigenMatrix() = input->EigenMatrix().transpose() * grad_output->EigenMatrix();
    }
    if (bias) {
        grad_bias->EigenVector() = grad_output->EigenMatrix().colwise().sum();
    }

    return {grad_input, grad_weight, grad_bias};
}
} // namespace infini_train::kernels::cpu

#define REGISTER_CPU_LINEAR_KERNEL(kernel_name)                                                                        \
    REGISTER_KERNEL(infini_train::DeviceType::kCPU, kernel_name, infini_train::kernels::cpu::kernel_name)

REGISTER_CPU_LINEAR_KERNEL(MatmulForward)
REGISTER_CPU_LINEAR_KERNEL(MatmulBackward)
REGISTER_CPU_LINEAR_KERNEL(LinearForward)
REGISTER_CPU_LINEAR_KERNEL(LinearBackward)

#undef REGISTER_CPU_LINEAR_KERNEL
