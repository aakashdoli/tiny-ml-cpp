#include "tinyml/linear_regression.hpp"

#include <stdexcept>

namespace tinyml {

LinearRegression::LinearRegression(double learning_rate, int epochs)
    : learning_rate_(learning_rate), epochs_(epochs) {
    if (learning_rate <= 0.0) throw std::invalid_argument("learning_rate must be > 0");
    if (epochs <= 0) throw std::invalid_argument("epochs must be > 0");
}

void LinearRegression::fit(const Matrix& X, const Vector& y) {
    if (X.empty() || X.size() != y.size()) {
        throw std::invalid_argument("X and y must be non-empty and the same length");
    }

    const std::size_t n = X.size();
    const std::size_t d = X[0].size();
    weights_.assign(d, 0.0);
    bias_ = 0.0;
    loss_history_.clear();

    for (int epoch = 0; epoch < epochs_; ++epoch) {
        Vector grad_w(d, 0.0);
        double grad_b = 0.0;
        double loss = 0.0;

        // 1. Look at every example and see how wrong we are
        for (std::size_t i = 0; i < n; ++i) {
            double error = predict(X[i]) - y[i];
            loss += error * error;
            for (std::size_t j = 0; j < d; ++j) grad_w[j] += error * X[i][j];
            grad_b += error;
        }

        // 2. Move each weight a small step in the direction that reduces the error
        for (std::size_t j = 0; j < d; ++j) {
            weights_[j] -= learning_rate_ * (2.0 / n) * grad_w[j];
        }
        bias_ -= learning_rate_ * (2.0 / n) * grad_b;

        loss_history_.push_back(loss / n);  // mean squared error
    }
}

double LinearRegression::predict(const Vector& row) const {
    if (row.size() != weights_.size()) {
        throw std::invalid_argument("Row has wrong number of features (did you call fit?)");
    }
    double result = bias_;
    for (std::size_t j = 0; j < row.size(); ++j) result += weights_[j] * row[j];
    return result;
}

Vector LinearRegression::predict(const Matrix& X) const {
    Vector out;
    out.reserve(X.size());
    for (const auto& row : X) out.push_back(predict(row));
    return out;
}

}  // namespace tinyml
