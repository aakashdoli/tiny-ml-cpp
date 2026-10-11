#pragma once

#include "tinyml/dataset.hpp"

namespace tinyml {

// Ways to measure how good a model is.

// Regression (predicting numbers)
double mean_squared_error(const Vector& y_true, const Vector& y_pred);
double root_mean_squared_error(const Vector& y_true, const Vector& y_pred);
double r2_score(const Vector& y_true, const Vector& y_pred);  // 1.0 = perfect

// Classification (predicting labels)
double accuracy(const Vector& y_true, const Vector& y_pred);  // share correct, 0..1

}  // namespace tinyml
