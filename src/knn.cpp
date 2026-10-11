#include "tinyml/knn.hpp"

#include <algorithm>
#include <map>
#include <stdexcept>
#include <utility>

namespace tinyml {

namespace {

// Straight-line distance, squared (we skip sqrt -- the order is the same)
double squared_distance(const Vector& a, const Vector& b) {
    double sum = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        double d = a[i] - b[i];
        sum += d * d;
    }
    return sum;
}

}  // namespace

KNNClassifier::KNNClassifier(int k) : k_(k) {
    if (k <= 0) throw std::invalid_argument("k must be > 0");
}

void KNNClassifier::fit(const Matrix& X, const Vector& y) {
    if (X.empty() || X.size() != y.size()) {
        throw std::invalid_argument("X and y must be non-empty and the same length");
    }
    X_train_ = X;
    y_train_ = y;
}

int KNNClassifier::predict(const Vector& row) const {
    if (X_train_.empty()) throw std::logic_error("Call fit() before predict()");

    // 1. Distance from the new point to every training point
    std::vector<std::pair<double, int>> dist;  // (distance, label)
    dist.reserve(X_train_.size());
    for (std::size_t i = 0; i < X_train_.size(); ++i) {
        dist.emplace_back(squared_distance(row, X_train_[i]),
                          static_cast<int>(y_train_[i]));
    }

    // 2. Keep only the k closest
    std::size_t k = std::min<std::size_t>(k_, dist.size());
    std::partial_sort(dist.begin(), dist.begin() + k, dist.end());

    // 3. Vote
    std::map<int, int> votes;
    for (std::size_t i = 0; i < k; ++i) votes[dist[i].second]++;

    int best_label = votes.begin()->first;
    int best_count = 0;
    for (const auto& [label, count] : votes) {
        if (count > best_count) {
            best_label = label;
            best_count = count;
        }
    }
    return best_label;
}

Vector KNNClassifier::predict(const Matrix& X) const {
    Vector out;
    out.reserve(X.size());
    for (const auto& row : X) out.push_back(predict(row));
    return out;
}

}  // namespace tinyml
