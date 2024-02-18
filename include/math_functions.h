#pragma once
#include <Eigen/Dense>


float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}
float sigmoid_derivative(float x) {
    return sigmoid(x) * (1 - sigmoid(x));
}

float relu(float x) {
    return (x > 0) ? x : 0;
}

float relu_derivative(float x) {
    return (x > 0) ? 1 : 0;
}

float _sgn(float x){
    if (x > 0) {
        return 1;
    } else if (x < 0){
        return -1;
    } else return 0;
}

std::function<float(float)> sgn(_sgn);

float stl_vec_mean(const std::vector<float>& vec){
    float sum = 0;
    for (auto v : vec){
        sum+=v;
    }
    return sum/vec.size();
}

float log_sum_exp(float a, float b) {
    if (a == b) {
        return a + std::log(2.0f);
    } else {
        float max_val = std::max(a, b);
        return max_val + std::log(std::exp(a - max_val) + std::exp(b - max_val));
    }
}

float safe_exp(float z) {
    float max_val = std::numeric_limits<float>::max();
    return (z > max_val) ? max_val : std::exp(z);
}

float stddev_vec(const Eigen::VectorXf& vec){
    float std_dev = std::sqrt((vec.array() - vec.mean()).square().sum()/(vec.size()-1));
    return std_dev;
}

float cross_entropy_loss(const Eigen::VectorXf& p, const Eigen::VectorXf& y){
    Eigen::VectorXf lnp = p.array().log();
    float loss = (y.array() * lnp.array()).sum();
    std::cout << "p:\n" << p << "\nln(p):\n" << lnp << std::endl;
    return loss;
}