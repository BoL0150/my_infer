#include <vector>
#include <cstdint>
#include <armadillo>
#pragma once
namespace my_infer {

template <class T>
class Tensor {
public:
    // 构造一个一维的tensor
    explicit Tensor(uint32_t size) ;
    // 构造二维张量
    explicit Tensor(uint32_t rows, uint32_t cols);
    // 构造三维张量
    explicit Tensor(uint32_t channels, uint32_t rows, uint32_t cols);
    explicit Tensor(const std::vector<uint32_t>& shapes);
    // 拷贝构造；注意！拷贝构造和移动构造一定要使用表达式初始化！否则和拷贝赋值运算符有什么区别？
    explicit Tensor(Tensor &t): data_(t.data()), raw_shape_(t.shapes()) {};
    // 移动构造
    explicit Tensor(Tensor &&t): data_(std::move(t.data())), raw_shape_(std::move(t.raw_shapes())) {};
    // 拷贝赋值，拷贝赋值运算符为了支持链式赋值，所以要返回当前对象的引用
    Tensor& operator=(Tensor &t);
    // 移动赋值
    Tensor& operator=(Tensor &&t);

    uint32_t rows() const;
    uint32_t cols() const;
    uint32_t channels() const;
    uint32_t size() const;
    uint32_t plane_size() const;
    bool empty() const;
    // 返回第channel通道中的数据
    const arma::fmat& slice(uint32_t channel) const;
    arma::fmat& slice(uint32_t channel);
    // 返回指定位置的元素
    T at(uint32_t channel, uint32_t row, uint32_t col) const;
    // 返回指定偏移量处的元素
    T& index(uint32_t offset);
    T index(uint32_t offset) const;
    

    // 使用value值初始化张量（广播）
    void Fill(T value);
    // 使用values数组中的值初始化张量，row_major表明values数组是行优先存储的还是列优先存储的
    void Fill(const std::vector<T> values, bool row_major = true);
    // 打印Tensor
    void Show() const;
    void Rand();
    // 对张量中的每个元素都进行操作
    void Transform(const std::function<T(T)>& filter);
    // 对张量进行reshape，注意！按照行优先reshape和按照列优先reshape是不一样的
    // 因为reshape本质上是把原来的张量展开成一维的，然后按照行优先或者列优先的方式
    // 存储到新的张量中
    void Reshape(const std::vector<uint32_t>& shapes, bool row_major = false);
    // 将tensor的数据按照行优先或者列优先存储成一维数组返回
    std::vector<T> values(bool row_major = true);

    std::vector<uint32_t> shapes() const;
    std::vector<uint32_t> &raw_shapes();
    arma::Cube<T>& data();
    // 将多维的tensor按照行优先或者列优先展平成一维的tensor
    void Flatten(bool row_major = false);
    void Padding(const std::vector<uint32_t>& pads, T padding_value);
private:
    // 使用vector来表示tensor的shape
    std::vector<uint32_t> raw_shape_;
    // 使用armadillo矩阵库作为tensor的底层实现，因为它提供了矩阵运算等操作
    // Tensor类相当于是对armadillo矩阵库各种操作的一个包装
    // 但是armadillo矩阵库是列优先，而各框架库中的Tensor是行优先
    arma::Cube<T> data_; 

};

}
