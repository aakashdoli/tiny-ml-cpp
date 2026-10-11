# tiny-ml-cpp

A small machine learning library written from scratch in modern C++17. No external libraries — just the standard library.

It implements two classic algorithms, plus the tools around them, to show how ML works under the hood:

- **Linear regression** trained with gradient descent (predicts numbers)
- **k-nearest neighbors** classifier (predicts labels)
- CSV loading, train/test split, feature scaling
- Metrics: MSE, RMSE, R², accuracy

## Results

**House prices** (linear regression, 200 houses, 20% held out for testing):

```
Training error (MSE): start 18532382.2 -> end 62248.3
Test R^2: 0.979   (1.0 = perfect)
```

**Point classification** (kNN, 3 groups, 300 points, 25% held out):

```
k =  1  ->  94.7%
k =  3  ->  96.0%
k =  9  ->  96.0%
```

## Build and run

You need a C++17 compiler and CMake 3.16+.

```bash
cmake -S . -B build
cmake --build build

./build/house_prices      # linear regression demo
./build/classify_points   # kNN demo
cd build && ctest         # run the tests
```

Run the demos from the project root so they find the `data/` folder, or pass a CSV path: `./build/house_prices my_data.csv`

## Use it in your own code

```cpp
#include "tinyml/dataset.hpp"
#include "tinyml/linear_regression.hpp"
#include "tinyml/metrics.hpp"

using namespace tinyml;

Dataset data = load_csv("data/houses.csv");   // last column = target
Dataset train, test;
train_test_split(data, 0.2, 42, train, test);

Standardizer scaler;
scaler.fit(train.X);                           // learn scaling from train only

LinearRegression model(0.05, 500);
model.fit(scaler.transform(train.X), train.y);

Vector pred = model.predict(scaler.transform(test.X));
double r2 = r2_score(test.y, pred);
```

## How it works

**Linear regression.** The model predicts `bias + w1*x1 + w2*x2 + ...`. It starts with all weights at zero, measures the average squared error, computes which direction makes the error smaller (the gradient), and takes a small step that way. This repeats for a set number of epochs.

**k-nearest neighbors.** To label a new point, it finds the `k` closest training points (Euclidean distance) and returns the most common label among them.

**Feature scaling.** Features like size (30–160 m²) and rooms (1–7) are on very different scales. `Standardizer` rescales each to mean 0 and standard deviation 1, so no single feature dominates. It is fit on training data only, to avoid leaking test information.

## Project structure

```
include/tinyml/   public headers
src/              implementations
examples/         two runnable demos
tests/            unit tests (no framework, run via ctest)
data/             sample datasets
```

## Data

The datasets in `data/` are synthetic, generated for this demo:

- `houses.csv`: size, rooms, age → price (thousand SEK), with added noise
- `points.csv`: 2D points from 3 overlapping clusters

## Possible next steps

- Logistic regression for classification
- Mini-batch gradient descent
- k-fold cross-validation
- Speed up kNN with a k-d tree
