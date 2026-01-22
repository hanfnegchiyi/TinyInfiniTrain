#include "cublas_v2.h"
#include "glog/logging.h"
#include <cub/block/block_reduce.cuh>

#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::kernels::cuda {

#define CUDA_CHECK(call)                                                                                               \
    do {                                                                                                               \
        cudaError_t status = call;                                                                                     \
        if (status != cudaSuccess) {                                                                                   \
            LOG(FATAL) << "CUDA Error: " << cudaGetErrorString(status) << " at " << __FILE__ << ":" << __LINE__;       \
        }                                                                                                              \
    } while (0)

#define CUBLAS_CHECK(call)                                                                                             \
    do {                                                                                                               \
        cublasStatus_t status = call;                                                                                  \
        if (status != CUBLAS_STATUS_SUCCESS) {                                                                         \
            LOG(FATAL) << "CUBLAS Error: " << cublasGetStatusString(status) << " at " << __FILE__ << ":" << __LINE__;  \
        }                                                                                                              \
    } while (0)

// 批量矩阵乘法kernel
__global__ void batched_matmul_forward_kernel(
    const float* A, const float* B, float* C,
    int M, int N, int K,
    int batch_size) {
    
    // 使用3D grid: x对应列，y对应行，z对应batch
    int batch = blockIdx.z;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (batch < batch_size && row < M && col < N) {
        // 计算当前batch的偏移
        int A_offset = batch * M * K;
        int B_offset = batch * K * N;
        int C_offset = batch * M * N;
        
        float sum = 0.0f;
        for (int k_idx = 0; k_idx < K; ++k_idx) {
            int idx_A = A_offset + row * K + k_idx;
            int idx_B = B_offset + k_idx * N + col;
            sum += A[idx_A] * B[idx_B];
        }
        
        int idx_C = C_offset + row * N + col;
        C[idx_C] = sum;
    }
}

// 支持广播的批量矩阵乘法kernel
__global__ void batched_matmul_broadcast_kernel(
    const float* A, const float* B, float* C,
    int M, int N, int K,
    int batch_size_A, int batch_size_B, int batch_size_C,
    int stride_A, int stride_B, int stride_C) {
    
    int batch = blockIdx.z;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N && batch < batch_size_C) {
        int batch_A = (batch_size_A == 1) ? 0 : batch;
        int batch_B = (batch_size_B == 1) ? 0 : batch;
        
        // 计算偏移
        int A_offset = batch_A * stride_A;
        int B_offset = batch_B * stride_B;
        int C_offset = batch * stride_C;
        
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            int idx_A = A_offset + row * K + k;
            int idx_B = B_offset + k * N + col;
            sum += A[idx_A] * B[idx_B];
        }
        
        int idx_C = C_offset + row * N + col;
        C[idx_C] = sum;
    }
}

// 批量矩阵乘法包装函数
void BatchedMatmulForwardCUDA(const float* A, const float* B, float* C,
                              int M, int N, int K, int batch_size, cudaStream_t stream) {
    
    // 设置3D网格
    dim3 blockSize(16, 16);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x,
                  (M + blockSize.y - 1) / blockSize.y,
                  batch_size);
    
    batched_matmul_forward_kernel<<<gridSize, blockSize, 0, stream>>>(
        A, B, C, M, N, K, batch_size);
}

// 支持广播的批量矩阵乘法
void BatchedMatmulBroadcastCUDA(const float* A, const float* B, float* C,
                                int M, int N, int K,
                                int batch_size_A, int batch_size_B, int batch_size_C,
                                cudaStream_t stream) {
    
    // 计算每个batch的步长
    int stride_A = M * K;
    int stride_B = K * N;
    int stride_C = M * N;
    
    dim3 blockSize(16, 16);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x,
                  (M + blockSize.y - 1) / blockSize.y,
                  batch_size_C);
    
    batched_matmul_broadcast_kernel<<<gridSize, blockSize, 0, stream>>>(
        A, B, C, M, N, K,
        batch_size_A, batch_size_B, batch_size_C,
        stride_A, stride_B, stride_C);
}

// 反向传播：计算dA = dC * B^T
__global__ void matmul_backward_dA_kernel(
    const float* dC, const float* B, float* dA,
    int M, int N, int K) {
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < K) {
        float sum = 0.0f;
        for (int n = 0; n < N; ++n) {
            int idx_dC = row * N + n;
            int idx_B = col * N + n;  // B^T的索引
            sum += dC[idx_dC] * B[idx_B];
        }
        
        int idx_dA = row * K + col;
        dA[idx_dA] = sum;
    }
}

// 反向传播：计算dB = A^T * dC
__global__ void matmul_backward_dB_kernel(
    const float* A, const float* dC, float* dB,
    int M, int N, int K) {
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < K && col < N) {
        float sum = 0.0f;
        for (int m = 0; m < M; ++m) {
            int idx_A = m * K + row;  // A^T的索引
            int idx_dC = m * N + col;
            sum += A[idx_A] * dC[idx_dC];
        }
        
        int idx_dB = row * N + col;
        dB[idx_dB] = sum;
    }
}

void MatmulBackwardCUDA(const float* A, const float* B, const float* dC,
                        float* dA, float* dB,
                        int M, int N, int K, cudaStream_t stream) {
    
    // 计算dA
    dim3 blockSize_dA(16, 16);
    dim3 gridSize_dA((K + blockSize_dA.x - 1) / blockSize_dA.x,
                     (M + blockSize_dA.y - 1) / blockSize_dA.y);
    matmul_backward_dA_kernel<<<gridSize_dA, blockSize_dA, 0, stream>>>(dC, B, dA, M, N, K);
    
    // 计算dB
    dim3 blockSize_dB(16, 16);
    dim3 gridSize_dB((N + blockSize_dB.x - 1) / blockSize_dB.x,
                     (K + blockSize_dB.y - 1) / blockSize_dB.y);
    matmul_backward_dB_kernel<<<gridSize_dB, blockSize_dB, 0, stream>>>(A, dC, dB, M, N, K);
}
std::shared_ptr<Tensor> MatmulForward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &other) {
    // =================================== 作业 ===================================
    // TODO：实现CUDA上的矩阵乘法前向计算
    // REF:
    // =================================== 作业 ===================================
    // 检查输入维度
    auto input_dims = input->Dims();
    auto other_dims = other->Dims();
    
    if (input_dims.size() < 2 || other_dims.size() < 2) {
        throw std::runtime_error("MatMul requires at least 2D tensors");
    }
    
    // 获取最后两个维度
    int64_t m = input_dims[input_dims.size() - 2];
    int64_t k = input_dims[input_dims.size() - 1];
    int64_t n = other_dims[other_dims.size() - 1];
    
    // 检查维度是否匹配
    if (k != other_dims[other_dims.size() - 2]) {
        throw std::runtime_error("Matrix dimensions do not match for multiplication");
    }
    
    // 计算输出维度
    std::vector<int64_t> output_dims;
    // 处理广播：比较除最后两个维度外的所有维度
    size_t num_batch_dims = std::max(input_dims.size(), other_dims.size()) - 2;
    for (size_t i = 0; i < num_batch_dims; ++i) {
        int64_t dim1 = (i < input_dims.size() - 2) ? input_dims[i] : 1;
        int64_t dim2 = (i < other_dims.size() - 2) ? other_dims[i] : 1;
        if (dim1 != dim2 && dim1 != 1 && dim2 != 1) {
            throw std::runtime_error("Batch dimensions are not broadcastable");
        }
        output_dims.push_back(std::max(dim1, dim2));
    }
    output_dims.push_back(m);
    output_dims.push_back(n);
    
    // 计算batch size（所有batch维度的乘积）
    int64_t batch_size = 1;
    for (size_t i = 0; i < output_dims.size() - 2; ++i) {
        batch_size *= output_dims[i];
    }
    
    // 创建输出Tensor
    auto output = std::make_shared<Tensor>(output_dims, input->Dtype(), input->GetDevice());
    
    // 获取CUDA设备
    auto device = input->GetDevice();
    if (device.Type() != DeviceType::kCUDA) {
        throw std::runtime_error("This function requires CUDA device");
    }

    // 获取数据指针
    float* input_data = static_cast<float*>(input->DataPtr());
    float* other_data = static_cast<float*>(other->DataPtr());
    float* output_data = static_cast<float*>(output->DataPtr());
    
    // 调用批量矩阵乘法kernel
    // 注意：需要实现BatchedMatmulForwardCUDA函数
    BatchedMatmulForwardCUDA(input_data, other_data, output_data,
                             static_cast<int>(m), static_cast<int>(n), static_cast<int>(k),
                             static_cast<int>(batch_size), nullptr);
    
    return output;
}

std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>
MatmulBackward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &other,
               const std::shared_ptr<Tensor> &grad_output) {
    // =================================== 作业 ===================================
    // TODO：实现CUDA上的矩阵乘法反向传播
    // REF:
    // =================================== 作业 ===================================

    auto device = input->GetDevice();
    if (device.Type() != DeviceType::kCUDA) {
        throw std::runtime_error("This function requires CUDA device");
    }

    auto input_dims = input->Dims();
    auto other_dims = other->Dims();

    int64_t m = input_dims[input_dims.size() - 2];
    int64_t k = input_dims[input_dims.size() - 1];
    int64_t n = other_dims[other_dims.size() - 1];
    // 创建梯度Tensor
    auto grad_input = std::make_shared<Tensor>(input->Dims(), input->Dtype(), device);
    auto grad_other = std::make_shared<Tensor>(other->Dims(), other->Dtype(), device);
    
    float* input_data = static_cast<float*>(input->DataPtr());
    float* other_data = static_cast<float*>(other->DataPtr());
    float* grad_output_data = static_cast<float*>(grad_output->DataPtr());
    float* grad_input_data = static_cast<float*>(grad_input->DataPtr());
    float* grad_other_data = static_cast<float*>(grad_other->DataPtr());
    
    // 直接调用CUDA函数
    MatmulBackwardCUDA(input_data, other_data, grad_output_data,
                       grad_input_data, grad_other_data,
                       static_cast<int>(m), static_cast<int>(n), static_cast<int>(k),
                       nullptr);  // 使用默认stream
    
    return {grad_input, grad_other};
}

__global__ void BiasCopyKernel(float *output, const float *bias, int bs, int out_features) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= bs * out_features) {
        return;
    }
    int j = idx % out_features;
    output[idx] = bias[j];
}

std::shared_ptr<Tensor> LinearForward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &weight,
                                      bool transpose, const std::shared_ptr<Tensor> &bias) {

    /*
        !transpose: output = input * weight + bias
        output[*, out_features] = input[*, in_features] * weight[in_features, out_features] + bias[out_features]

        transpose:  output = input * weight^T + bias
        output[*, out_features] = input[*, in_features] * weight[out_features, in_features]^T + bias[out_features]
    */

    const auto &input_dims = input->Dims();
    CHECK_GE(input_dims.size(), 2);
    const int64_t bs = std::accumulate(input_dims.rbegin() + 1, input_dims.rend(), 1, std::multiplies<int64_t>{});
    const int64_t in_features = *input_dims.rbegin();

    const auto &weight_dims = weight->Dims();
    CHECK_EQ(weight_dims.size(), 2);
    CHECK_EQ(in_features, weight_dims[transpose ? 1 : 0]);

    // As for cublas:
    // C = alpha * op(B) * op(A) + beta * C
    // Dimensions:
    //   input:  (bs, in_features)
    //   weight: (in_features, out_features) or (out_features, in_features) if transposed
    //   output: (bs, out_features)
    const int64_t out_features = weight_dims[transpose ? 0 : 1];

    auto output_dims = input_dims;
    *output_dims.rbegin() = out_features;
    auto output = std::make_shared<Tensor>(output_dims, DataType::kFLOAT32, input->GetDevice());

    if (bias) {
        CHECK_EQ(bias->Dims().size(), 1);
        CHECK_EQ(bias->Dims()[0], out_features);
        int threads_per_block = 256;
        int num_blocks = (bs * out_features + threads_per_block - 1) / threads_per_block;
        BiasCopyKernel<<<num_blocks, threads_per_block>>>(
            static_cast<float *>(output->DataPtr()), static_cast<const float *>(bias->DataPtr()), bs, out_features);
    } else {
        output->Fill<float>(0.0f);
    }

    const float alpha = 1.0f;
    const float beta = 1.0f;
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    if (transpose) {
        // weight is [out_features, in_features] here

        // output = input * weight.T --> output.T = weight * input.T
        // C = output.T[out_features, bs]
        // A = weight.T[in_features, out_features]
        // B = input.T[in_features, bs]
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, out_features, bs, in_features, &alpha,
                                 static_cast<const float *>(weight->DataPtr()), in_features,
                                 static_cast<const float *>(input->DataPtr()), in_features, &beta,
                                 static_cast<float *>(output->DataPtr()), out_features));
    } else {
        // output = input * weight --> output.T =  weight.T * input.T
        // C = output.T[out_features, bs]
        // A = weight.T[out_features, in_features]
        // B = input.T[in_features, bs]
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, out_features, bs, in_features, &alpha,
                                 static_cast<const float *>(weight->DataPtr()), out_features,
                                 static_cast<const float *>(input->DataPtr()), in_features, &beta,
                                 static_cast<float *>(output->DataPtr()), out_features));
    }
    CUBLAS_CHECK(cublasDestroy(handle));
    return output;
}

template <int BLOCK_SIZE>
__global__ void ReduceColumnsKernel(const float *__restrict__ input, float *__restrict__ output, int num_rows,
                                    int num_cols) {
    using BlockReduce = cub::BlockReduce<float, BLOCK_SIZE>;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    int row = blockIdx.x;
    float sum = 0.0f;

    for (int col = threadIdx.x; col < num_cols; col += blockDim.x) { sum += input[row * num_cols + col]; }

    float reduced = BlockReduce(temp_storage).Sum(sum);

    if (threadIdx.x == 0) {
        output[row] = reduced;
    }
}

std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>
LinearBackward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &weight, bool transpose,
               int64_t out_features, const std::shared_ptr<Tensor> &grad_output, const bool bias) {
    const auto &input_dims = input->Dims();
    CHECK_GE(input_dims.size(), 2);
    const int64_t bs = std::accumulate(input_dims.rbegin() + 1, input_dims.rend(), 1, std::multiplies<int64_t>{});
    const int64_t in_features = *input_dims.rbegin();

    const auto &weight_dims = weight->Dims();
    CHECK_EQ(weight_dims.size(), 2);
    CHECK_EQ(in_features, weight_dims[transpose ? 1 : 0]);
    CHECK_EQ(out_features, weight_dims[transpose ? 0 : 1]);

    auto grad_input = std::make_shared<Tensor>(input_dims, DataType::kFLOAT32, grad_output->GetDevice());
    auto grad_weight = std::make_shared<Tensor>(weight_dims, DataType::kFLOAT32, grad_output->GetDevice());
    grad_input->Fill<float>(0.0f);
    grad_weight->Fill<float>(0.0f);
    std::shared_ptr<Tensor> grad_bias = nullptr;
    if (bias) {
        grad_bias = std::make_shared<Tensor>(std::vector<int64_t>{out_features}, DataType::kFLOAT32,
                                             grad_output->GetDevice());
        grad_bias->Fill<float>(0.0f);
    }

    float alpha = 1.0f;
    float beta = 0.0f;
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    if (transpose) {
        // weight is [out_features, in_features] here

        // d_input = d_output * weight --> d_input.T = weight.T * d_output.T
        // C = d_input.T[in_features, bs]
        // A = weight.T[in_features, out_features]
        // B = d_output.T[out_features, bs]
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, in_features, bs, out_features, &alpha,
                                 static_cast<const float *>(weight->DataPtr()), in_features,
                                 static_cast<const float *>(grad_output->DataPtr()), out_features, &beta,
                                 static_cast<float *>(grad_input->DataPtr()), in_features));

        // d_weight = d_output.T * input --> d_weight.T = input.T * d_output
        // C = d_weight.T[in_features, out_features]
        // A = input.T[in_features, bs]
        // B = d_output.T[out_features, bs]
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, in_features, out_features, bs, &alpha,
                                 static_cast<const float *>(input->DataPtr()), in_features,
                                 static_cast<const float *>(grad_output->DataPtr()), out_features, &beta,
                                 static_cast<float *>(grad_weight->DataPtr()), in_features));
    } else {
        // weight is [in_features, out_features] here

        // d_input = d_output * weight.T --> d_input.T = weight * d_output.T
        // C = d_input.T[in_features, bs]
        // A = weight.T[out_features, in_features]
        // B = d_output.T[out_features, bs]
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, in_features, bs, out_features, &alpha,
                                 static_cast<const float *>(weight->DataPtr()), out_features,
                                 static_cast<const float *>(grad_output->DataPtr()), out_features, &beta,
                                 static_cast<float *>(grad_input->DataPtr()), in_features));

        // d_weight = input.T * d_output --> d_weight.T = d_output.T * input
        // C = d_weight.T[out_features, in_features]
        // A = d_output.T[out_features, bs]
        // B = input.T[in_features, bs]
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, out_features, in_features, bs, &alpha,
                                 static_cast<const float *>(grad_output->DataPtr()), out_features,
                                 static_cast<const float *>(input->DataPtr()), in_features, &beta,
                                 static_cast<float *>(grad_weight->DataPtr()), out_features));
    }

    // d_bias = \sum_i(i=0, bs-1) d_output[i]
    if (bias) {
        constexpr int BLOCK_SIZE = 256;
        int threads_per_block = BLOCK_SIZE;
        int num_blocks = out_features;
        ReduceColumnsKernel<BLOCK_SIZE>
            <<<num_blocks, threads_per_block>>>(static_cast<const float *>(grad_output->DataPtr()),
                                                static_cast<float *>(grad_bias->DataPtr()), out_features, bs);
    }

    CUBLAS_CHECK(cublasDestroy(handle));

    return {grad_input, grad_weight, grad_bias};
}
} // namespace infini_train::kernels::cuda

#define REGISTER_CUDA_LINEAR_KERNEL(kernel_name)                                                                       \
    REGISTER_KERNEL(infini_train::DeviceType::kCUDA, kernel_name, infini_train::kernels::cuda::kernel_name)

REGISTER_CUDA_LINEAR_KERNEL(MatmulForward)
REGISTER_CUDA_LINEAR_KERNEL(MatmulBackward)
REGISTER_CUDA_LINEAR_KERNEL(LinearForward)
REGISTER_CUDA_LINEAR_KERNEL(LinearBackward)

#undef REGISTER_CUDA_LINEAR_KERNEL
