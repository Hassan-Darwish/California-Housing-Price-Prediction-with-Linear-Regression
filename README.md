# Linear Regression for Housing Price Prediction

This project implements a linear regression model from scratch in Python to predict median housing prices using the California Housing dataset. The entire machine learning workflow is covered, from data cleaning and feature engineering to model training and evaluation against a benchmark.

The project emphasizes a foundational understanding of machine learning algorithms and data preprocessing techniques.

📌 Project Overview
This project is organized into two main parts, contained within separate Jupyter notebooks:

**data_cleaning.ipynb:** Loads the raw dataset, performs comprehensive cleaning, feature scaling, categorical encoding, and strategic splitting to create robust training and testing sets.

**training.ipynb:** Implements the linear regression algorithm using NumPy, trains the model with gradient descent, and evaluates its performance on the test data.

📂 Project Structure

```plaintext
project-root/
├── data_cleaning.ipynb          # Notebook for data preprocessing and splitting
├── training.ipynb               # Notebook for model implementation, training, and evaluation
├── housing.csv                  # Raw input dataset
├── housing_training.csv         # Generated training set
└── housing_testing.csv          # Generated testing set
```

🎯 Key Steps & Techniques

### Data Cleaning & Feature Engineering

* **Data Inspection:** Loaded the raw housing.csv dataset and used .info() to identify missing values in the total_bedrooms column.
* **Normalization:** Applied Z-score normalization to all numerical features to standardize the data scale.
* **Categorical Encoding:** Converted the categorical ocean_proximity feature into numerical format using one-hot encoding.
* **Feature Creation:** Engineered an income_cat feature based on median_income to facilitate a more representative data split.
* **Stratified Sampling:** Used StratifiedShuffleSplit on the income_cat feature to create training and test sets that are balanced with respect to income levels, preventing sampling bias.

### Model Implementation & Training

* **Algorithm from Scratch:** Implemented a linear regression model using only NumPy for mathematical operations.
* **Optimization:** Used the gradient descent algorithm to iteratively train the model and find the optimal weights and bias.
* **Benchmarking:** Compared the custom model's performance against the standard LinearRegression model from Scikit-learn to validate the implementation.

📊 Evaluation & Results
The model was evaluated on the unseen test set using Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R-squared (R²) metrics. The custom model performed nearly identically to the Scikit-learn benchmark, confirming its correctness.

**Test Set Performance (Custom Model):**

* MSE: 0.353

* RMSE: 0.594

* R²: 0.639

Below is a plot comparing the model's predictions to the true values on the test set.

![True vs Predicted Values](<img width="565" height="453" alt="image" src="https://github.com/user-attachments/assets/094fa508-26a8-4499-b38a-b45f51c4501d" />
){width="640" alt="True vs Predicted Values"}

🚀 How to Run

**Prerequisites**

* Python 3.x
* Jupyter Notebook
* Libraries: pandas, numpy, scikit-learn, matplotlib

**Instructions**

1. Clone the repository to your local machine.

```bash
git clone https://github.com/Hassan-Darwish/California-Housing-Price-Prediction-with-Linear-Regression
```

2. Run the data_cleaning.ipynb notebook first to generate the housing_training.csv and housing_testing.csv files.

3. Run the training.ipynb notebook to train the linear regression model and view the performance evaluation.

🛠️ Future Enhancements

* Implement more advanced regression models like Random Forest or Gradient Boosting.
* Incorporate cross-validation for more robust model evaluation.
* Perform hyperparameter tuning to optimize model performance.

📜 License

MIT License

👤 Author

Developed by Hassan Darwish
