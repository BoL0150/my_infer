#include "data/tensor.hpp"
#include <memory>

using namespace my_infer;
// 注意！对于模版函数,由于需要在编译的时候就生成函数，所以要将函数的定义和实现写在一起！
template<class T>
std::shared_ptr<Tensor<T>> TensorCreate(uint32_t size) {
    return std::make_shared<Tensor<T>>(size);
}

template<class T>
std::shared_ptr<Tensor<T>> TensorCreate(uint32_t rows, uint32_t cols) {
    return std::make_shared<Tensor<T>>(rows, cols);
}

template<class T>
std::shared_ptr<Tensor<T>> TensorCreate(uint32_t channels, uint32_t rows, uint32_t cols) {
    return std::make_shared<Tensor<T>>(channels, rows, cols);
}
