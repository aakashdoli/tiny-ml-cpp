// Classify 2D points into 3 groups using k-nearest neighbors,
// and compare different values of k.
//
// Usage: ./classify_points [path/to/points.csv]

#include <iomanip>
#include <iostream>

#include "tinyml/dataset.hpp"
#include "tinyml/knn.hpp"
#include "tinyml/metrics.hpp"

using namespace tinyml;

int main(int argc, char** argv) {
    std::string path = argc > 1 ? argv[1] : "data/points.csv";

    try {
        Dataset data = load_csv(path);
        std::cout << "Loaded " << data.size() << " points from " << path << "\n";

        Dataset train, test;
        train_test_split(data, 0.25, /*seed=*/7, train, test);

        Standardizer scaler;
        scaler.fit(train.X);
        Matrix X_train = scaler.transform(train.X);
        Matrix X_test = scaler.transform(test.X);

        std::cout << "\nAccuracy on " << test.size() << " unseen points:\n";
        std::cout << std::fixed << std::setprecision(1);
        for (int k : {1, 3, 5, 9, 15}) {
            KNNClassifier model(k);
            model.fit(X_train, train.y);
            double acc = accuracy(test.y, model.predict(X_test));
            std::cout << "  k = " << std::setw(2) << k << "  ->  " << acc * 100 << "%\n";
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
