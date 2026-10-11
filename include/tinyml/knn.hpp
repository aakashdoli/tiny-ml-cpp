#pragma once

#include "tinyml/dataset.hpp"

namespace tinyml {

// k-Nearest Neighbors: to classify a new point, find the k training points
// closest to it and let them vote. The most common label wins.
// There is no real "training" -- it just remembers the data.
class KNNClassifier {
public:
    explicit KNNClassifier(int k = 3);

    void fit(const Matrix& X, const Vector& y);
    int predict(const Vector& row) const;
    Vector predict(const Matrix& X) const;

private:
    int k_;
    Matrix X_train_;
    Vector y_train_;
};

}  // namespace tinyml
