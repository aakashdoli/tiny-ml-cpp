#pragma once

#include "tinyml/dataset.hpp"

namespace tinyml {

// Predicts a number (like a house price) as:
//   prediction = bias + w1*x1 + w2*x2 + ...
// It learns the weights with gradient descent: start with guesses,
// measure the error, nudge the weights to make the error smaller, repeat.
class LinearRegression {
public:
    LinearRegression(double learning_rate = 0.01, int epochs = 1000);

    void fit(const Matrix& X, const Vector& y);
    double predict(const Vector& row) const;
    Vector predict(const Matrix& X) const;

    const Vector& weights() const { return weights_; }
    double bias() const { return bias_; }
    const Vector& loss_history() const { return loss_history_; }

private:
    double learning_rate_;
    int epochs_;
    Vector weights_;
    double bias_ = 0.0;
    Vector loss_history_;  // error after each epoch, to see it going down
};

}  // namespace tinyml
