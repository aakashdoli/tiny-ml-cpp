#pragma once

#include <string>
#include <vector>

namespace tinyml {

// One row = one example. Each row has some features (X) and one answer (y).
using Matrix = std::vector<std::vector<double>>;
using Vector = std::vector<double>;

struct Dataset {
    std::vector<std::string> feature_names;
    Matrix X;  // features, one row per example
    Vector y;  // the answer for each row

    std::size_t size() const { return X.size(); }
};

// Load a CSV file with a header row. The LAST column is the answer (y).
Dataset load_csv(const std::string& path);

// Shuffle the data and split it into a training part and a test part.
// test_ratio = 0.2 means 20% goes to testing.
void train_test_split(const Dataset& data, double test_ratio, unsigned seed,
                      Dataset& train, Dataset& test);

// Puts every feature on the same scale (mean 0, std 1).
// Without this, a big feature like "size in m2" would drown out "rooms".
class Standardizer {
public:
    void fit(const Matrix& X);                     // learn mean and std from training data
    Matrix transform(const Matrix& X) const;       // apply them
    Vector transform_row(const Vector& row) const;

private:
    Vector mean_;
    Vector std_;
};

}  // namespace tinyml
