// Predict house prices from size, rooms and age using linear regression.
//
// Usage: ./house_prices [path/to/houses.csv]

#include <iomanip>
#include <iostream>

#include "tinyml/dataset.hpp"
#include "tinyml/linear_regression.hpp"
#include "tinyml/metrics.hpp"

using namespace tinyml;

int main(int argc, char** argv) {
    std::string path = argc > 1 ? argv[1] : "data/houses.csv";

    try {
        // 1. Load data
        Dataset data = load_csv(path);
        std::cout << "Loaded " << data.size() << " houses from " << path << "\n";

        // 2. Split: 80% train, 20% test
        Dataset train, test;
        train_test_split(data, 0.2, /*seed=*/42, train, test);

        // 3. Scale features (learned from TRAIN only, so test data stays unseen)
        Standardizer scaler;
        scaler.fit(train.X);
        Matrix X_train = scaler.transform(train.X);
        Matrix X_test = scaler.transform(test.X);

        // 4. Train
        LinearRegression model(/*learning_rate=*/0.05, /*epochs=*/500);
        model.fit(X_train, train.y);

        const auto& loss = model.loss_history();
        std::cout << std::fixed << std::setprecision(1);
        std::cout << "Training error (MSE): start " << loss.front()
                  << " -> end " << loss.back() << "\n";

        // 5. Test on unseen houses
        Vector pred = model.predict(X_test);
        std::cout << std::setprecision(3);
        std::cout << "\nTest results on " << test.size() << " unseen houses:\n";
        std::cout << "  RMSE : " << root_mean_squared_error(test.y, pred)
                  << " (thousand SEK, average miss)\n";
        std::cout << "  R^2  : " << r2_score(test.y, pred) << " (1.0 = perfect)\n";

        // 6. Use the model on a brand new house
        Vector new_house = {85, 3, 12};  // 85 m2, 3 rooms, 12 years old
        double price = model.predict(scaler.transform_row(new_house));
        std::cout << std::setprecision(0);
        std::cout << "\nNew house (85 m2, 3 rooms, 12 years): ~" << price
                  << " thousand SEK\n";
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
