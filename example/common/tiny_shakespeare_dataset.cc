#include "example/common/tiny_shakespeare_dataset.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/tensor.h"

namespace {
using DataType = infini_train::DataType;
using TinyShakespeareType = TinyShakespeareDataset::TinyShakespeareType;
using TinyShakespeareFile = TinyShakespeareDataset::TinyShakespeareFile;

const std::unordered_map<int, TinyShakespeareType> kTypeMap = {
    {20240520, TinyShakespeareType::kUINT16}, // GPT-2
    {20240801, TinyShakespeareType::kUINT32}, // LLaMA 3
};

const std::unordered_map<TinyShakespeareType, size_t> kTypeToSize = {
    {TinyShakespeareType::kUINT16, 2},
    {TinyShakespeareType::kUINT32, 4},
};
 
const std::unordered_map<TinyShakespeareType, DataType> kTypeToDataType = {
    {TinyShakespeareType::kUINT16, DataType::kUINT16},
    {TinyShakespeareType::kUINT32, DataType::kINT32},
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

TinyShakespeareFile ReadTinyShakespeareFile(const std::string &path, size_t sequence_length) {
    /* =================================== 作业 ===================================
       TODO：实现二进制数据集文件解析
       文件格式说明：
    ----------------------------------------------------------------------------------
    | HEADER (1024 bytes)                     | DATA (tokens)                        |
    | magic(4B) | version(4B) | num_toks(4B) | reserved(1012B) | token数据           |
    ----------------------------------------------------------------------------------
       =================================== 作业 =================================== */
    // 打开文件
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs.is_open()) {
        LOG(FATAL) << "Cannot open dataset file: " << path;
    }
    
    // 读取头部文件
    std::vector<uint8_t> header_bytes = ReadSeveralBytesFromIfstream(1024, &ifs);
    
    // 解析头部信息
    uint32_t magic_number = BytesToType<uint32_t>(header_bytes, 0);
    uint32_t version = BytesToType<uint32_t>(header_bytes, 4);
    uint32_t num_tokens = BytesToType<uint32_t>(header_bytes, 8);
    
    // 根据magic_number确定数据类型
    auto type_it = kTypeMap.find(magic_number);
    if (type_it == kTypeMap.end()) {
        LOG(FATAL) << "Unknown magic number: " << magic_number;
    }
    TinyShakespeareType data_type = type_it->second;
    
    // 获取数据类型大小和对应的DataType
    size_t type_size = kTypeToSize.at(data_type);
    DataType tensor_data_type = kTypeToDataType.at(data_type);
    
    // 计算需要读取的数据字节数
    size_t data_size_bytes = num_tokens * type_size;
    
    // 读取token数据
    std::vector<uint8_t> data_bytes = ReadSeveralBytesFromIfstream(data_size_bytes, &ifs);
    
    // 创建Tensor
    // [num_tokens]
    std::vector<int64_t> tensor_dims = {static_cast<int64_t>(num_tokens)};
    infini_train::Tensor tensor(tensor_dims, DataType::kINT64);
    int64_t* tensor_data = static_cast<int64_t*>(tensor.DataPtr());
    
    // 创建并返回TinyShakespeareFile结构体
    TinyShakespeareFile result;
    result.type = data_type;
    result.dims = tensor_dims;
    result.tensor = std::move(tensor);  // 使用移动语义
    
    return result;
}
} // namespace

TinyShakespeareDataset::TinyShakespeareDataset(const std::string &filepath, size_t sequence_length) {
    // =================================== 作业 ===================================
    // TODO：初始化数据集实例
    // HINT: 调用ReadTinyShakespeareFile加载数据文件
    // =================================== 作业 ===================================
    // 读取数据集文件
    text_file_ = ReadTinyShakespeareFile(filepath, sequence_length);
    
    // 获取数据类型大小
    size_t type_size = kTypeToSize.at(text_file_.type);

    // 获取总token数
    uint32_t num_tokens = text_file_.dims[0];
    
    // 计算样本数量
    const_cast<size_t&>(sequence_size_in_bytes_) = sequence_length * type_size * 2;
    const_cast<size_t&>(num_samples_) = (num_tokens - 1) / sequence_length;
    
    // 验证数据是否存在
    if (num_samples_ == 0) {
        LOG(FATAL) << "Dataset too small for sequence length " << sequence_length 
                   << ". Need at least " << (sequence_length + 1) << " tokens.";
    }

}

std::pair<std::shared_ptr<infini_train::Tensor>, std::shared_ptr<infini_train::Tensor>>
TinyShakespeareDataset::operator[](size_t idx) const {
    CHECK_LT(idx, text_file_.dims[0] - 1);
    std::vector<int64_t> dims = std::vector<int64_t>(text_file_.dims.begin() + 1, text_file_.dims.end());
    // x: (seq_len), y: (seq_len) -> stack -> (bs, seq_len) (bs, seq_len)
    return {std::make_shared<infini_train::Tensor>(text_file_.tensor, idx * sequence_size_in_bytes_, dims),
            std::make_shared<infini_train::Tensor>(text_file_.tensor, idx * sequence_size_in_bytes_ + sizeof(int64_t),
                                                   dims)};
}

size_t TinyShakespeareDataset::Size() const { return num_samples_; }
