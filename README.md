# Heavy Equipment Selling Price Prediction Challenge

Explain the data using statistics and predict the selling prices of heavy equipment using classical Machine Learning models.

## Overview

The Heavy Equipment Selling Price Prediction Challenge focuses on exploring a dataset containing operational, transactional, geographical, and technical information about heavy industrial machinery and equipment.

Each record represents a finalized accounting or operational event and includes information such as system metadata, geographic indicators, component specifications, and mechanical configurations.

The objective is to perform exploratory data analysis, identify meaningful patterns in numerical and categorical variables, and develop robust Machine Learning models to predict the target variable, `TargetValue`.

## Project Objectives

- Perform exploratory data analysis (EDA) to understand the dataset.
- Summarize numerical variables using descriptive statistics.
- Analyze categorical and high-cardinality technical features.
- Identify missing values, outliers, distributions, and relationships between features.
- Apply suitable data preprocessing and feature engineering techniques.
- Train and evaluate classical Machine Learning regression models.
- Compare model performance and select an effective predictive model.
- Generate predictions for the test dataset.
- Prepare a submission file in the required format.

## Dataset Description

The dataset contains information about heavy equipment transactions and associated technical and operational characteristics.

The main files used for this challenge are:

- `train.csv` — Training data containing input features and the target variable.
- `test.csv` — Test data containing input features for which predictions must be generated.
- `sample_submission.csv` — Sample file illustrating the required submission format.

### Target Variable

`TargetValue` represents the value to be predicted for each transaction.

### Identifier

`TransactionID` uniquely identifies each transaction and is required in the final submission file.

## Exploratory Data Analysis

The statistical analysis and exploratory data analysis cover the following areas:

- Dataset dimensions and data types.
- Descriptive statistics for numerical variables.
- Frequency distributions of categorical variables.
- Missing-value analysis.
- Distribution of the target variable.
- Outlier detection.
- Relationships between numerical features and the target.
- Analysis of high-cardinality categorical and technical string features.
- Identification of potentially useful predictive features.

## Data Preprocessing and Feature Engineering

The preprocessing pipeline may include:

- Handling missing values.
- Encoding categorical variables.
- Processing high-cardinality features.
- Converting suitable columns to appropriate data types.
- Creating additional informative features.
- Applying transformations to skewed numerical variables.
- Preparing consistent training and test feature matrices.
- Preventing data leakage during model training and evaluation.

Preprocessing steps should be fitted using the training data and applied consistently to validation and test data.

## Machine Learning Models

Classical Machine Learning regression algorithms can be trained and compared, including:

- Ridge Regression
- Extra Trees Regressor
- Histogram-based Gradient Boosting Regressor
- LightGBM Regressor
- XGBoost Regressor
- CatBoost Regressor

The final model selection should be based on validation performance and the competition's evaluation metric.

## Model Evaluation

The competition evaluates submissions using **Root Mean Squared Logarithmic Error (RMSLE)**.

RMSLE measures the difference between the logarithms of the predicted and actual target values, making it useful when relative prediction errors matter across a wide range of equipment prices.

The metric is defined as:

\[
\mathrm{RMSLE}
=
\sqrt{
\frac{1}{n}
\sum_{i=1}^{n}
\left[
\ln(1+\hat{y}_i)-\ln(1+y_i)
\right]^2
}
\]

Where:

- \(n\) is the number of observations.
- \(y_i\) is the actual target value.
- \(\hat{y}_i\) is the predicted target value.

Lower RMSLE values indicate better predictive performance.

Since logarithmic evaluation requires non-negative target values and predictions, predictions must be checked for invalid negative values before submission.

## Submission Format

The final submission file must be a CSV file containing the `TransactionID` and `TargetValue` columns.

Example:

| TransactionID | TargetValue |
|---:|---:|
| 1139307 | 44674.9 |
| 1139419 | 91450.1 |
| 1139482 | 76618.1 |
| 1139522 | 75195.6 |
| 1139684 | 8340.3 |

The submission must:

- Include the column headers.
- Contain one prediction for every transaction in the test dataset.
- Preserve the original `TransactionID` values and their corresponding row order.
- Use `TargetValue` as the prediction column name.
- Be saved in CSV format.

The submission file can be generated using pandas:

```python
submission = pd.DataFrame({
    "TransactionID": test["TransactionID"],
    "TargetValue": predictions
})

submission.to_csv("submission.csv", index=False)
```

Here, `predictions` should contain the final model's predictions for the test dataset.

## Expected Outcome

The expected outcome is a complete Machine Learning workflow that combines statistical analysis, data preprocessing, feature engineering, model training, model comparison, and prediction generation.

The project aims to identify patterns in heavy equipment transaction data and build a reliable regression model capable of predicting equipment selling prices on unseen data.


- [Kaggle Competition](https://www.kaggle.com/competitions/144271)
- [Scikit-learn RMSLE Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.root_mean_squared_log_error.html)
