#include "tinyml/metrics.hpp"

#include <cmath>
#include <stdexcept>

namespace tinyml {

namespace {
void check(const Vector& a, const Vector& b) {
    if (a.empty() || a.size() != b.size()) {
        throw std::invalid_argument("Vectors must be non-empty and the same length");
    }
}
}  // namespace

double mean_squared_error(const Vector& y_true, const Vector& y_pred) {
    check(y_true, y_pred);
    double sum = 0.0;
    for (std::size_t i = 0; i < y_true.size(); ++i) {
        double e = y_true[i] - y_pred[i];
        sum += e * e;
    }
    return sum / y_true.size();
}

double root_mean_squared_error(const Vector& y_true, const Vector& y_pred) {
    return std::sqrt(mean_squared_error(y_true, y_pred));
}

double r2_score(const Vector& y_true, const Vector& y_pred) {
    check(y_true, y_pred);
    double mean = 0.0;
    for (double v : y_true) mean += v;
    mean /= y_true.size();

    double ss_res = 0.0, ss_tot = 0.0;
    for (std::size_t i = 0; i < y_true.size(); ++i) {
        ss_res += (y_true[i] - y_pred[i]) * (y_true[i] - y_pred[i]);
        ss_tot += (y_true[i] - mean) * (y_true[i] - mean);
    }
    if (ss_tot == 0.0) return ss_res == 0.0 ? 1.0 : 0.0;
    return 1.0 - ss_res / ss_tot;
}

double accuracy(const Vector& y_true, const Vector& y_pred) {
    check(y_true, y_pred);
    std::size_t correct = 0;
    for (std::size_t i = 0; i < y_true.size(); ++i) {
        if (y_true[i] == y_pred[i]) ++correct;
    }
    return static_cast<double>(correct) / y_true.size();
}

}  // namespace tinyml
