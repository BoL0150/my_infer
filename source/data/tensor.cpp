#include <iostream>
#include "data/tensor.hpp"
#include <cassert>
// #include <glog/logging.h>
namespace my_infer {

template<class T>
Tensor<T>::Tensor(uint32_t size) {
    // 初始化一个cube，如果张量是一维的，那么cube的row和channel都为1，col等于size
    this->data_ = arma::fcube(1, size, 1);
    // 初始化shape，shape是一个vector；如果张量是一维的，那么shape长度为1
    this->raw_shape_ = std::vector<uint32_t>{size};
}
template<class T>
Tensor<T>::Tensor(uint32_t rows, uint32_t cols) {
    this->data_ = arma::fcube(rows, cols, 1);
    this->raw_shape_ = std::vector<uint32_t>{rows, cols};
}

template<class T>
Tensor<T>::Tensor(uint32_t channels, uint32_t rows, uint32_t cols) {
    this->data_ = arma::fcube(rows, cols, channels);
    // 这里没有使用张量前面维度为1会让shape退化的特性
    this->raw_shape_ = std::vector<uint32_t>{rows, cols, channels};
}

// 不管shape是什么样的，作为fcube的data_都是三维的，所以即使不存在channel维度或者row维度，他们的值仍然是1
template<class T>
uint32_t Tensor<T>::rows() const {
    assert(!this->data_.empty());
    return data_.n_rows;
}

template<class T>
uint32_t Tensor<T>::cols() const {
    assert(!this->data_.empty());
    return data_.n_cols;
}
template<class T>
uint32_t Tensor<T>::channels() const {
    assert(!this->data_.empty());
    return data_.n_slices;
}
template<class T>
uint32_t Tensor<T>::size() const {
    assert(!this->data_.empty());
    return data_.size();
}
template<class T>
const arma::fmat& Tensor<T>::slice(uint32_t channel) const {
    assert(channel < this->channels());
    return data_.slice(channel);
}
template<class T>
arma::fmat& Tensor<T>::slice(uint32_t channel) {
    assert(channel < this->channels());
    return data_.slice(channel);
}
template<class T>
T Tensor<T>::at(uint32_t channel, uint32_t row, uint32_t col) const {
    assert(channel < channels() && row < rows() && col < cols());
    return data_.at(row, col, channel);
}
template<class T>
const T& Tensor<T>::index(uint32_t offset) {
    assert(offset >= 0 && offset < size());
    return data_.at(offset);
}
template<class T>
T Tensor<T>::index(uint32_t offset) const {
    assert(offset >= 0 && offset < size());
    return data_.at(offset);
}
template<class T>
void Tensor<T>::Fill(T value) {
    assert(!data_.empty());
    data_.fill(value);
}
template<class T>
void Tensor<T>::Fill(const std::vector<T> values, bool row_major) {
    assert(values.size() == size());
    // 如果输入的矩阵的values数组是按照行优先存储的，那么需要将它先转换成列优先再复制给fmat
    // 行主序和列主序只是影响一个channel内部的存储顺序，不影响channel的顺序
    if (row_major && raw_shape_.size() >= 2) {
        uint32_t channel_num = raw_shape_.size() == 3 ? channels() : 1;
        uint32_t plane_size = rows() * cols();
        for (int i = 0; i < channel_num; i++) {
            arma::fmat &channel_data = slice(i);
            const arma::fmat &input_data = 
                arma::fmat(values.data() + i * plane_size, cols(), rows());
            channel_data = input_data.t();
        }
    } else {
        // 如果输入的矩阵的values数组是按照列优先存储的，那么直接将它复制给fmat即可，因为fmat也是列优先存储的
        // 或者输入的是row_major并且为为1也可以直接复制
        std::copy(values.begin(), values.end(), data_.memptr());
    }
}
template<class T>
std::vector<T> Tensor<T>::values(bool row_major) {
    std::vector<float> result(size());
    const uint32_t plane_size = cols() * rows();
    for (int i = 0; i < channels(); i++) {
        arma::fmat &plane = slice(i);
        if (row_major) plane = plane.t();
        std::copy(plane.memptr() + i * plane_size, plane.memptr() + (i + 1) * plane_size, result.data() + i * plane_size);
    }
    return result;
}
template<class T>
void Tensor<T>::Show() const {
    assert(size() != 0);
    for (int i = 0; i < channels(); i++) {
        std::cout << "Channel: " << i << std::endl;
        std::cout << slice(i) << std::endl;
    }
}
template<class T>
void Tensor<T>::Rand() {
    assert(size() != 0);
    data_.randn(); 
}
template<class T>
void Tensor<T>::Transform(const std::function<T(T)>& filter) {
    assert(size() != 0);
    data_.transform(filter);
}
template<class T>
void Tensor<T>::Reshape(const std::vector<uint32_t>& shapes, bool row_major) {
    uint32_t new_size = 1;
    for (int i = 0; i < shapes.size(); i++) {
        new_size *= shapes[i];
    }
    assert(new_size == size());

}
template<class T>
std::vector<uint32_t> Tensor<T>::shapes() const {
    return raw_shape_;
}
template<class T>
const std::vector<uint32_t> & Tensor<T>::raw_shapes() {
    return raw_shape_;
}

}
