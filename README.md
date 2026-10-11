# tiny-ml-cpp

A zero-dependency machine learning library and interactive visualizer written from scratch in modern **C++17** (Standard Library only).

It implements fundamental machine learning algorithms, preprocessing pipelines, and evaluation metrics without third-party frameworks like PyTorch or Scikit-learn, demonstrating how ML works under the hood.

---

## Interactive Web Dashboard & Visualizer

An interactive dark-mode dashboard is included to explore predictions, inspect convergence, test 2D spatial clustering, and trigger compiled C++ binaries from the browser.

```bash
# Launch the web visualizer
python3 server.py

# Open in your browser:
# http://localhost:8080
```

---

### 1. Linear Regression & House Price Valuation

![Linear Regression Dashboard](docs/screenshots/dashboard_regression.png)

#### What every feature in this view tells you:

* **Top Metric Strip**:
  * **Linear Regression $R^2$ (0.979 / 1.0)**: Measures the proportion of variance in house prices explained by the features. A score of $0.979$ indicates an exceptional fit on unseen testing data.
  * **kNN Accuracy (96.0%)**: Generalization score achieved on held-out points across $k \in [3..15]$.
  * **Unit Test Suite (6 / 6 Passing)**: Status of self-contained mathematical and algorithm verification tests.
  * **C++ Latency (< 2 ms)**: Native execution speed compiled with `-O3` optimizations.
* **Interactive House Price Predictor (Left Panel)**:
  * **Living Area Slider (30–180 m²)**: Modifies house square footage. Larger area exerts the highest positive weight impact on property value ($w_1 \approx +1464.6$).
  * **Number of Rooms Slider (1–7 rooms)**: Adds incremental value for room utility ($w_2 \approx +181.9$).
  * **Building Age Slider (0–80 years)**: Applies property depreciation. Older structures reduce valuation ($w_3 \approx -307.3$).
  * **Estimated Valuation Card**: Displays real-time predicted price in thousand SEK (e.g. `~4,025 thousand SEK` for an 85 m², 3-room, 12-year-old house), alongside currency approximations.
  * **Weight Impact Breakdown**: Shows exact value contributions added or subtracted by each feature after standardization ($w_j \cdot z_j$).
  * **Mathematical Formulation**: Exposes the learned linear model:
    $$\hat{y} = \text{bias} + \sum_{j=1}^{d} w_j \cdot z_j, \quad z_j = \frac{x_j - \mu_j}{\sigma_j}$$
* **Visualizations & Convergence (Right Panel)**:
  * **Gradient Descent Loss Curve**: Charts Mean Squared Error (MSE) dropping by **99.7%** from an initial guess error of $18,779,935.4$ down to $62,171.2$ across 500 training epochs.
  * **Size vs. Price Regression Scatter Plot**: Plots real data points from `data/houses.csv` against the learned regression line, with real-time crosshair tracking the active slider parameters.

---

### 2. k-Nearest Neighbors (2D Cluster Classification)

![k-Nearest Neighbors Dashboard](docs/screenshots/dashboard_knn.png)

#### What every feature in this view tells you:

* **Interactive 2D Cluster Canvas (Left Panel)**:
  * **300 Colored Data Points**: Renders the complete synthetic dataset from `data/points.csv` divided into 3 clusters:
    * **Class 0 (Cluster A, Emerald)**: Lower-left cluster centered around $(1.5, 2.5)$.
    * **Class 1 (Cluster B, Cyan)**: Lower-right cluster centered around $(5.5, 2.8)$.
    * **Class 2 (Cluster C, Violet)**: Upper cluster centered around $(3.8, 7.2)$.
  * **Interactive Test Probe (Amber Circle)**: **Click or drag anywhere** on the canvas to inspect classification.
  * **Euclidean Distance Vectors**: Draws laser projection lines connecting the probe directly to its $k$ closest neighbors in real-time.
  * **Radius Boundary Circle**: Shows the maximum distance sphere encompassing the $k$-th closest neighbor.
* **kNN Parameters & Vote Tally (Right Panel)**:
  * **Neighbor Count ($k$) Slider**: Adjusts $k$ from 1 to 21. Lower $k$ creates sensitive local boundaries; higher $k$ provides smoother, noise-resistant boundaries.
  * **Current Classification Card**: Displays the plurality winner and percentage breakdown of neighbor votes.
  * **Vote Progress Bars**: Real-time bars showing vote counts per class for the current probe location.
  * **Accuracy Benchmark Across $k$**: Compares accuracy on 75 unseen test points ($k=1 \rightarrow 89.3\%$, $k=3..15 \rightarrow 96.0\%$).

---

### 3. Live C++ Execution Terminal

![Live C++ Execution Terminal](docs/screenshots/dashboard_terminal.png)

#### What every feature in this view tells you:

* **Native Execution Buttons**:
  * **`Run ./build/house_prices`**: Executes the compiled regression binary via the Python backend bridge, displaying training loss history, test RMSE, $R^2$, and single-house prediction.
  * **`Run ./build/classify_points`**: Executes the compiled kNN binary, showing accuracy benchmarks on unseen test points.
  * **`Run ./build/tinyml_tests`**: Executes the native unit test suite.
* **Terminal Window**:
  * Replicates authentic macOS dark terminal styling with colored syntax highlighting (`PASS`/`FAIL`, metrics, commands).
  * **Latency Indicator**: Measures and displays backend round-trip execution latency (typically 5–10 ms).
  * **Copy Button**: Copies raw terminal output directly to the clipboard.

---

### 4. Datasets Explorer

![Datasets Explorer](docs/screenshots/dashboard_data.png)

#### What every feature in this view tells you:

* **Dataset Switcher**: Toggle between `data/houses.csv` (200 records) and `data/points.csv` (300 records).
* **Live Search Filter**: Instantly filters tabular rows by numerical value or feature name.
* **Scrollable Table**: Inspect raw features and targets used during model training.

---

## How it Works Under the Hood

### 1. Linear Regression (Gradient Descent)
The model predicts:
$$\hat{y} = b + w_1 x_1 + w_2 x_2 + \dots + w_d x_d$$

It initializes weights to 0 and minimizes Mean Squared Error (MSE) using batch gradient descent:
$$\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (\hat{y}_i - y_i)^2$$

At each epoch, weights are nudged in the opposite direction of the gradient:
$$w_j \leftarrow w_j - \alpha \frac{2}{n} \sum_{i=1}^{n} (\hat{y}_i - y_i) x_{i,j}, \quad b \leftarrow b - \alpha \frac{2}{n} \sum_{i=1}^{n} (\hat{y}_i - y_i)$$

### 2. k-Nearest Neighbors (kNN)
To classify a new sample $\mathbf{x}_{\text{query}}$:
1. Calculates squared Euclidean distance to all training points:
   $$d(\mathbf{x}, \mathbf{y})^2 = \sum_{j=1}^{d} (x_j - y_j)^2$$
2. Extracts top $k$ nearest neighbors using `std::partial_sort` (avoiding the cost of full sorting).
3. Evaluates a majority vote among neighbors; the most frequent label wins.

### 3. Feature Scaling (`Standardizer`)
Features on disparate scales (e.g., area $30–180 \text{ m}^2$ vs. rooms $1–7$) can distort gradients. `Standardizer` rescales each feature to mean 0 and standard deviation 1:
$$z = \frac{x - \mu}{\sigma}$$
*Fit exclusively on training data* to prevent test data leakage.

---

## Build and Run CLI

### Prerequisites
A modern C++17 compiler (`clang++` or `g++`).

### Option A: Direct Compilation (No CMake Required)
```bash
mkdir -p build
clang++ -std=c++17 -O3 -Iinclude src/*.cpp examples/house_prices.cpp -o build/house_prices
clang++ -std=c++17 -O3 -Iinclude src/*.cpp examples/classify_points.cpp -o build/classify_points
clang++ -std=c++17 -O3 -Iinclude src/*.cpp tests/test_main.cpp -o build/tinyml_tests

# Run demos
./build/house_prices
./build/classify_points

# Run test suite
./build/tinyml_tests
```

### Option B: CMake
```bash
cmake -S . -B build
cmake --build build
cd build && ctest --output-on-failure
```

---

## Using in Your Own Code

```cpp
#include "tinyml/dataset.hpp"
#include "tinyml/linear_regression.hpp"
#include "tinyml/metrics.hpp"

using namespace tinyml;

// 1. Load CSV (last column is target y)
Dataset data = load_csv("data/houses.csv");

// 2. Train / test split (80% train, 20% test, seeded RNG)
Dataset train, test;
train_test_split(data, 0.2, 42, train, test);

// 3. Scale features (learn from train only)
Standardizer scaler;
scaler.fit(train.X);

// 4. Train Linear Regression
LinearRegression model(/*learning_rate=*/0.05, /*epochs=*/500);
model.fit(scaler.transform(train.X), train.y);

// 5. Evaluate on unseen test data
Vector pred = model.predict(scaler.transform(test.X));
double r2 = r2_score(test.y, pred);
double rmse = root_mean_squared_error(test.y, pred);
```

---

## Project Structure

```
tiny-ml-cpp/
├── CMakeLists.txt              # CMake build configuration
├── server.py                   # Local web server & C++ binary bridge
├── ui/                         # Interactive web dashboard
│   ├── index.html              # Dashboard markup
│   ├── style.css               # Modern dark theme styles
│   └── app.js                  # Canvases, sliders & terminal logic
├── include/tinyml/             # Public header files
│   ├── dataset.hpp             # Dataset, CSV parser, splitter, Standardizer
│   ├── linear_regression.hpp   # Linear regression model
│   ├── knn.hpp                 # k-Nearest Neighbors classifier
│   └── metrics.hpp             # MSE, RMSE, R², accuracy
├── src/                        # Implementations
│   ├── dataset.cpp
│   ├── linear_regression.cpp
│   ├── knn.cpp
│   └── metrics.cpp
├── examples/                   # Runnable CLI demos
│   ├── house_prices.cpp        # Linear regression demo
│   └── classify_points.cpp     # kNN classification demo
├── tests/
│   └── test_main.cpp           # Self-contained unit tests
├── data/                       # Synthetic CSV datasets
│   ├── houses.csv              # Size, rooms, age -> price
│   └── points.csv              # 2D coordinates -> class label
└── docs/screenshots/           # High-resolution dashboard screenshots
```

---

## License

MIT License. Open source and educational.
