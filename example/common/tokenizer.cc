#include "example/common/tokenizer.h"

#include <cctype>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

#include "glog/logging.h"

namespace infini_train {

constexpr uint32_t kGpt2Eot = 50256;
constexpr uint32_t kLLaMA3Eot = 128001;
constexpr uint64_t kRandomU32Multiplier = 0x2545F4914F6CDD1Dull;
constexpr float kF32Divisor = 16777216.0f; // 2^24
constexpr uint64_t kRngState = 1337;

using Version = Tokenizer::Version;

const std::unordered_map<uint32_t, uint32_t> kEotMap = {
    {20240328, kGpt2Eot},   // GPT-2
    {20240801, kLLaMA3Eot}, // LLaMA-3
};

const std::unordered_map<uint32_t, std::vector<uint32_t>> kPromptMap = {
    // e.g. "The meaning of life is"
    // ref: https://tiktokenizer.vercel.app/
    {20240328, std::vector<uint32_t>{464, 3616, 286, 1204, 318}}, // GPT-2
    {20240801, std::vector<uint32_t>{791, 7438, 315, 2324, 374}}, // LLaMA-3
};

std::vector<uint8_t> ReadSeveralBytesFromIfstream(size_t num_bytes, std::ifstream *ifs) {
    std::vector<uint8_t> result(num_bytes);
    ifs->read(reinterpret_cast<char *>(result.data()), num_bytes);
    return result;
}

template <typename T> T BytesToType(const std::vector<uint8_t> &bytes, size_t offset) {
    static_assert(std::is_trivially_copyable<T>::value, "T must be trivially copyable.");
    T value;
    std::memcpy(&value, &bytes[offset], sizeof(T));
    return value;
}

unsigned int RandomU32(uint64_t &state) {
    state ^= state >> 12;
    state ^= state << 25;
    state ^= state >> 27;
    return (state * kRandomU32Multiplier) >> 32;
}

float RandomF32(uint64_t &state) { // random float32 in [0,1)
    return (RandomU32(state) >> 8) / kF32Divisor;
}

int SampleMult(float *probabilities, int n, float coin) {
    // sample index from probabilities (they must sum to 1!)
    // coin is a random number in [0, 1), usually from RandomF32()
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += probabilities[i];
        if (coin < cdf) {
            return i;
        }
    }
    return n - 1; // in case of rounding errors
}

Tokenizer::Tokenizer(const std::string &filepath) {
    /* ===================================== 作业 =====================================
    TODO：实现Tokenizer二进制文件加载

    文件格式说明：
    ----------------------------------------------------------------------------------
    | HEADER (1024 bytes)                     | VOCAB TABLE                           |
    | magic(4B) | version(4B) | vocab_size(4B) | reserved(1012B) | token词表数据       |
    ----------------------------------------------------------------------------------
    ===================================== 作业 ===================================== */
    std::ifstream ifs(filepath, std::ios::binary);
    std::vector<uint8_t> header_bytes = ReadSeveralBytesFromIfstream(1024,&ifs);
    magic_number_ = BytesToType<uint32_t>(header_bytes,0);
    uint32_t version_data = BytesToType<uint32_t>(header_bytes,4);
    vocab_size_ = BytesToType<uint32_t>(header_bytes,8);

    auto it = kEotMap.find(magic_number_);
    eot_token_ = it->second;

    token_table_.reserve(vocab_size_);
    for(int i = 0; i < vocab_size_;i++){
        //读取当前token字节长度
        uint8_t token_len = 0;
        ifs.read(reinterpret_cast<char *>(&token_len), 1);
        std::vector<uint8_t> token_bytes = ReadSeveralBytesFromIfstream(token_len,&ifs);
        std::string token(token_bytes.begin(), token_bytes.end());

        token_table_.push_back(token);
    }
}

std::string Tokenizer::Decode(uint32_t token_id) const {
    /* ===================================== 作业 =====================================
    TODO：实现token_id到文本的转换
    功能描述：根据token_id返回对应的文本片段
    ===================================== 作业 ===================================== */
    return token_table_[token_id];
}

void Tokenizer::GenerateText(infini_train::nn::Module &model, uint32_t batch_size, uint32_t sequence_length,
                             uint32_t text_length, Device device) const {
    std::vector<int64_t> dims;
    dims.assign({batch_size, sequence_length});
    // x_tensor (FLAGS_batch_size, FLAGS_sequence_length) eq:(4, 64)
    infini_train::Tensor x_tensor = infini_train::Tensor(dims, DataType::kINT64);
    int64_t *x_buff = static_cast<int64_t *>(x_tensor.DataPtr());
    for (int i = 0; i < batch_size * sequence_length; ++i) { x_buff[i] = eot_token_; }

    // Give some contexts: "The meaning of life is "
    auto prompt = kPromptMap.at(magic_number_);
    auto prompt_len = prompt.size();
    for (int i = 0; i < prompt_len; ++i) { x_buff[i] = prompt[i]; }
    std::cout << "The meaning of life is";

    auto x = std::make_shared<infini_train::Tensor>(x_tensor.To(device));
    uint64_t kRngState = kRngState;
    LOG(INFO) << "start generate text:";
    for (int t = prompt_len; t < text_length; t++) {
        /* ===================================== 作业 =====================================
        TODO：实现单步文本生成逻辑
        HINT：调用model.Forward推理获取logits，根据推理结果进行随机采样，调用Decode获取文本结果
        ===================================== 作业 ===================================== */
        // x        [batch,sequence_len]
        //result    [batch,sequence_len,vocab_size]
        std::vector<std::shared_ptr<infini_train::Tensor>> result = model.Forward({x});
        auto logits_tensor = result[0];
        
        // 2. 获取logits形状
        auto logits_shape = logits_tensor->Dims();
        int64_t vocab_size = logits_shape[2];  // 从tensor获取vocab_size
        
        // 3. 准备概率数组
        std::vector<float> probabilities(vocab_size);
        
        // 4. 处理每个batch
        for (int b = 0; b < batch_size; b++) {
            // 计算当前batch最后一个位置的logits起始偏移
            int64_t batch_offset = b * sequence_length * vocab_size;
            int64_t pos_offset = (sequence_length - 1) * vocab_size;
            int64_t start_idx = batch_offset + pos_offset;
            
            // 获取logits指针
            float* logits_ptr = static_cast<float*>(logits_tensor->DataPtr()) + start_idx;
            
            // 5. 计算softmax
            float max_logit = -std::numeric_limits<float>::max();
            for (int v = 0; v < vocab_size; v++) {
                if (logits_ptr[v] > max_logit) {
                    max_logit = logits_ptr[v];
                }
            }
            
            float sum_exp = 0.0f;
            for (int v = 0; v < vocab_size; v++) {
                float exp_val = std::exp(logits_ptr[v] - max_logit);
                probabilities[v] = exp_val;
                sum_exp += exp_val;
            }
            
            for (int v = 0; v < vocab_size; v++) {
                probabilities[v] /= sum_exp;
            }
            
            // 6. 采样下一个token
            float random_val = RandomF32(kRngState);
            int next_token_id = SampleMult(probabilities.data(), vocab_size, random_val);
            
            // 7. 更新当前batch的输入序列
            int64_t *batch_x_buff = x_buff + b * sequence_length;
            for (int i = 0; i < sequence_length - 1; i++) {
                batch_x_buff[i] = batch_x_buff[i + 1];
            }
            batch_x_buff[sequence_length - 1] = next_token_id;
            
            // 8. 输出第一个batch的token
            if (b == 0) {
                std::string next_token_text = Decode(next_token_id);
                std::cout << next_token_text;
                std::cout.flush();
            }
        }
        
        // 9. 更新x张量用于下一次迭代
        x = std::make_shared<infini_train::Tensor>(x_tensor.To(device));

    }
    std::cout << std::endl;
}
} // namespace infini_train
