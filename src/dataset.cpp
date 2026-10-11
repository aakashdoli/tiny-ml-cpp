#include "tinyml/dataset.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>

namespace tinyml {

namespace {

std::vector<std::string> split_line(const std::string& line) {
    std::vector<std::string> parts;
    std::stringstream ss(line);
    std::string cell;
    while (std::getline(ss, cell, ',')) {
        parts.push_back(cell);
    }
    return parts;
}

}  // namespace

Dataset load_csv(const std::string& path) {
    std::ifstream file(path);
    if (!file) {
        throw std::runtime_error("Could not open file: " + path);
    }

    Dataset data;
    std::string line;

    // Header row: all columns except the last are feature names
    if (!std::getline(file, line)) {
        throw std::runtime_error("File is empty: " + path);
    }
    auto header = split_line(line);
    if (header.size() < 2) {
        throw std::runtime_error("Need at least one feature and one target column");
    }
    data.feature_names.assign(header.begin(), header.end() - 1);

    int line_number = 1;
    while (std::getline(file, line)) {
        ++line_number;
        if (line.empty()) continue;

        auto cells = split_line(line);
        if (cells.size() != header.size()) {
            throw std::runtime_error("Wrong number of columns on line " +
                                     std::to_string(line_number));
        }

        Vector row;
        for (std::size_t i = 0; i + 1 < cells.size(); ++i) {
            row.push_back(std::stod(cells[i]));
        }
        data.X.push_back(row);
        data.y.push_back(std::stod(cells.back()));
    }

    if (data.X.empty()) {
        throw std::runtime_error("No data rows in: " + path);
    }
    return data;
}

void train_test_split(const Dataset& data, double test_ratio, unsigned seed,
                      Dataset& train, Dataset& test) {
    if (test_ratio <= 0.0 || test_ratio >= 1.0) {
        throw std::invalid_argument("test_ratio must be between 0 and 1");
    }

    // Make a list of row numbers 0..n-1 and shuffle it
    std::vector<std::size_t> idx(data.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::mt19937 rng(seed);
    std::shuffle(idx.begin(), idx.end(), rng);

    std::size_t n_test = static_cast<std::size_t>(data.size() * test_ratio);

    train = Dataset{data.feature_names, {}, {}};
    test = Dataset{data.feature_names, {}, {}};
    for (std::size_t i = 0; i < idx.size(); ++i) {
        Dataset& target = (i < n_test) ? test : train;
        target.X.push_back(data.X[idx[i]]);
        target.y.push_back(data.y[idx[i]]);
    }
}

void Standardizer::fit(const Matrix& X) {
    if (X.empty()) throw std::invalid_argument("Cannot fit on empty data");

    std::size_t n = X.size();
    std::size_t d = X[0].size();
    mean_.assign(d, 0.0);
    std_.assign(d, 0.0);

    for (const auto& row : X)
        for (std::size_t j = 0; j < d; ++j) mean_[j] += row[j];
    for (auto& m : mean_) m /= n;

    for (const auto& row : X)
        for (std::size_t j = 0; j < d; ++j) {
            double diff = row[j] - mean_[j];
            std_[j] += diff * diff;
        }
    for (auto& s : std_) {
        s = std::sqrt(s / n);
        if (s == 0.0) s = 1.0;  // a column that never changes: avoid dividing by 0
    }
}

Vector Standardizer::transform_row(const Vector& row) const {
    Vector out(row.size());
    for (std::size_t j = 0; j < row.size(); ++j) {
        out[j] = (row[j] - mean_[j]) / std_[j];
    }
    return out;
}

Matrix Standardizer::transform(const Matrix& X) const {
    Matrix out;
    out.reserve(X.size());
    for (const auto& row : X) out.push_back(transform_row(row));
    return out;
}

}  // namespace tinyml
