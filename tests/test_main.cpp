// Small test suite with no external framework.
// Each test prints PASS or FAIL. The program returns 1 if anything fails.

#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

#include "tinyml/dataset.hpp"
#include "tinyml/knn.hpp"
#include "tinyml/linear_regression.hpp"
#include "tinyml/metrics.hpp"

using namespace tinyml;

static int failures = 0;

#define CHECK(cond)                                                          \
    do {                                                                     \
        if (!(cond)) {                                                       \
            std::cerr << "  FAIL: " #cond " (line " << __LINE__ << ")\n";    \
            ++failures;                                                      \
        }                                                                    \
    } while (0)

static bool near(double a, double b, double tol = 1e-3) { return std::fabs(a - b) < tol; }

void test_linear_regression_learns_a_line() {
    // y = 3x + 2, no noise. The model should find w=3, b=2.
    Matrix X;
    Vector y;
    for (int i = 0; i < 20; ++i) {
        double x = i / 10.0;
        X.push_back({x});
        y.push_back(3 * x + 2);
    }
    LinearRegression model(0.1, 5000);
    model.fit(X, y);
    CHECK(near(model.weights()[0], 3.0, 1e-2));
    CHECK(near(model.bias(), 2.0, 1e-2));
    CHECK(model.loss_history().back() < model.loss_history().front());
}

void test_knn_simple_groups() {
    Matrix X = {{0, 0}, {0, 1}, {1, 0}, {10, 10}, {10, 11}, {11, 10}};
    Vector y = {0, 0, 0, 1, 1, 1};
    KNNClassifier model(3);
    model.fit(X, y);
    CHECK(model.predict(Vector{0.5, 0.5}) == 0);
    CHECK(model.predict(Vector{10.5, 10.5}) == 1);
}

void test_metrics() {
    Vector a = {1, 2, 3};
    CHECK(near(mean_squared_error(a, {1, 2, 3}), 0.0));
    CHECK(near(mean_squared_error(a, {2, 3, 4}), 1.0));
    CHECK(near(r2_score(a, a), 1.0));
    CHECK(near(accuracy({1, 0, 1, 1}, {1, 1, 1, 1}), 0.75));
}

void test_standardizer() {
    Matrix X = {{1, 100}, {2, 200}, {3, 300}};
    Standardizer s;
    s.fit(X);
    Matrix Z = s.transform(X);
    CHECK(near(Z[1][0], 0.0));  // middle value becomes 0
    CHECK(near(Z[1][1], 0.0));
    CHECK(near(Z[0][0], Z[0][1]));  // both columns end up on the same scale
}

void test_split_sizes() {
    Dataset d;
    for (int i = 0; i < 10; ++i) {
        d.X.push_back({double(i)});
        d.y.push_back(i);
    }
    Dataset train, test;
    train_test_split(d, 0.3, 1, train, test);
    CHECK(train.size() == 7);
    CHECK(test.size() == 3);
}

void test_csv_loading_and_errors() {
    const std::string path = "tinyml_test.csv";
    {
        std::ofstream f(path);
        f << "a,b,target\n1,2,3\n4,5,6\n";
    }
    Dataset d = load_csv(path);
    CHECK(d.size() == 2);
    CHECK(d.feature_names.size() == 2);
    CHECK(near(d.y[1], 6.0));
    std::remove(path.c_str());

    bool threw = false;
    try {
        load_csv("does_not_exist.csv");
    } catch (const std::runtime_error&) {
        threw = true;
    }
    CHECK(threw);
}

int main() {
    struct { const char* name; void (*fn)(); } tests[] = {
        {"linear regression learns a line", test_linear_regression_learns_a_line},
        {"knn separates simple groups", test_knn_simple_groups},
        {"metrics", test_metrics},
        {"standardizer", test_standardizer},
        {"train/test split sizes", test_split_sizes},
        {"csv loading and errors", test_csv_loading_and_errors},
    };

    for (auto& t : tests) {
        int before = failures;
        t.fn();
        std::cout << (failures == before ? "PASS  " : "FAIL  ") << t.name << "\n";
    }
    std::cout << (failures == 0 ? "\nAll tests passed.\n" : "\nSome tests failed.\n");
    return failures == 0 ? 0 : 1;
}
