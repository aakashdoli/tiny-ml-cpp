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
#include "tinyml/baseband/complex_sample.hpp"
#include "tinyml/baseband/fixed_vector.hpp"
#include "tinyml/baseband/lms_equalizer.hpp"
#include "tinyml/baseband/qam_demapper.hpp"
#include "tinyml/baseband/ring_buffer.hpp"

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

void test_fixed_vector() {
    using namespace tinyml::baseband;
    FixedVector<int, 4> v;
    CHECK(v.empty());
    CHECK(v.capacity() == 4);
    v.push_back(10);
    v.push_back(20);
    v.push_back(30);
    CHECK(v.size() == 3);
    CHECK(v[0] == 10);
    CHECK(v.back() == 30);
    v.pop_back();
    CHECK(v.size() == 2);
    v.clear();
    CHECK(v.empty());
}

void test_complex_and_q15() {
    using namespace tinyml::baseband;
    ComplexFloat a{3.0f, 4.0f};
    CHECK(near(a.mag_sq(), 25.0f));
    CHECK(near(a.mag(), 5.0f));

    ComplexFloat b{1.0f, -2.0f};
    // (3 + 4j)(1 - 2j) = (3*1 - 4*(-2)) + j(3*(-2) + 4*1) = (3 + 8) + j(-6 + 4) = 11 - 2j
    ComplexFloat prod = a * b;
    CHECK(near(prod.i, 11.0f));
    CHECK(near(prod.q, -2.0f));

    // Q15 conversion round-trip
    ComplexFloat norm_cf{0.5f, -0.25f};
    ComplexQ15 q15 = ComplexQ15::from_float(norm_cf);
    ComplexFloat recovered = q15.to_float();
    CHECK(near(recovered.i, 0.5f, 1e-4));
    CHECK(near(recovered.q, -0.25f, 1e-4));
}

void test_spsc_ring_buffer() {
    using namespace tinyml::baseband;
    SPSCRingBuffer<int, 4> rb;
    CHECK(rb.empty());
    CHECK(rb.push(100));
    CHECK(rb.push(200));
    CHECK(rb.push(300));
    CHECK(rb.push(400));
    CHECK(!rb.push(500)); // Full

    int val = 0;
    CHECK(rb.pop(val) && val == 100);
    CHECK(rb.pop(val) && val == 200);
    CHECK(rb.pop(val) && val == 300);
    CHECK(rb.pop(val) && val == 400);
    CHECK(!rb.pop(val)); // Empty
}

void test_qam16_demapper() {
    using namespace tinyml::baseband;
    // Every nibble from 0 to 15 must roundtrip cleanly through modulate -> demap
    for (uint8_t nibble = 0; nibble < 16; ++nibble) {
        ComplexFloat sym = QAM16::modulate(nibble);
        uint8_t decoded = QAM16::demap(sym);
        CHECK(decoded == nibble);
    }
}

void test_lms_equalizer_convergence() {
    using namespace tinyml::baseband;
    LMSEqualizer<5> eq(0.05f);
    // Transmit constant sequence through simple channel with echo h = [1.0, 0.3j]
    ComplexFloat tx{0.707f, 0.707f};
    ComplexFloat prev{0.0f, 0.0f};

    float initial_err = 0.0f;
    float final_err = 0.0f;

    for (int i = 0; i < 200; ++i) {
        ComplexFloat rx = tx + prev * ComplexFloat{0.0f, 0.3f};
        prev = tx;
        ComplexFloat out = eq.filter_and_adapt(rx, tx);
        float err = (tx - out).mag();
        if (i < 5) initial_err += err;
        if (i > 190) final_err += err;
    }
    CHECK(final_err < initial_err); // Must learn channel inversion
}

int main() {
    struct { const char* name; void (*fn)(); } tests[] = {
        {"linear regression learns a line", test_linear_regression_learns_a_line},
        {"knn separates simple groups", test_knn_simple_groups},
        {"metrics", test_metrics},
        {"standardizer", test_standardizer},
        {"train/test split sizes", test_split_sizes},
        {"csv loading and errors", test_csv_loading_and_errors},
        {"baseband: FixedVector zero-heap", test_fixed_vector},
        {"baseband: Complex & Q15 DSP types", test_complex_and_q15},
        {"baseband: SPSC lock-free ring buffer", test_spsc_ring_buffer},
        {"baseband: QAM-16 modulation & demapping", test_qam16_demapper},
        {"baseband: LMS adaptive channel equalizer", test_lms_equalizer_convergence},
    };

    for (auto& t : tests) {
        int before = failures;
        t.fn();
        std::cout << (failures == before ? "PASS  " : "FAIL  ") << t.name << "\n";
    }
    std::cout << (failures == 0 ? "\nAll tests passed.\n" : "\nSome tests failed.\n");
    return failures == 0 ? 0 : 1;
}
