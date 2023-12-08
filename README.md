# Data Preparation

## Description:
This code cell initializes training data for a simple machine learning model. It creates two NumPy arrays, `x_train` and `y_train`, to be used as features and corresponding labels, respectively.

## Variables:
- `x_train`: NumPy array containing the feature values `[1, 2, 3]`.
- `y_train`: NumPy array containing the corresponding label values `[500.0, 750.0, 1000.0]`.

## Usage:
This data can be used to train a regression model where `x_train` represents the input features, and `y_train` represents the corresponding output labels.


# Computing Cost Function

## Description:
This code cell defines a function `computing_cost` to calculate the cost of a linear regression model. The cost is computed using the mean squared error between the predicted values and the actual labels.

## Parameters:
- `x`: NumPy array representing the feature values (numpy.ndarray, shape: (m, ) where m is the number of samples).
- `y`: NumPy array representing the corresponding label values (numpy.ndarray, shape: (m, ) where m is the number of samples).
- `w`: Weight parameter of the linear regression model (float).
- `b`: Bias parameter of the linear regression model (float).

## Returns:
- `total_cost`: The computed cost using the mean squared error formula (float).

## Implementation Details:
The cost function (mean squared error) is given by the formula:

$$ J(w, b) = \frac{1}{2m} \sum_{i=1}^{m} (w \cdot x_i + b - y_i)^2 $$

where:
- \( J(w,b) \) is the cost function.
- \( m \) is the number of samples.
- \( w \) is the weight parameter.
- \( b \) is the bias parameter.
- \( x_i \) is the ith feature.
- \( y_i \) is the ith label.

The function iterates through each data point, computes the predicted value using the linear regression model, calculates the squared difference between the predicted and actual values, and accumulates these squared differences. Finally, it divides the total by twice the number of data points (\(2m\)) to obtain the mean squared error.

## Usage:
```python
cost = computing_cost(x_train, y_train, weight, bias)

# Vectorized Computing Cost Function

## Description:
This code cell defines a vectorized version of the `computing_cost` function, referred to as `computing_cost_vectorised`. The vectorized implementation takes advantage of NumPy operations to improve efficiency.

## Parameters:
- `x`: NumPy array representing the feature values (numpy.ndarray, shape: (m, ) where m is the number of samples).
- `y`: NumPy array representing the corresponding label values (numpy.ndarray, shape: (m, ) where m is the number of samples).
- `w`: Weight parameter of the linear regression model (float).
- `b`: Bias parameter of the linear regression model (float).

## Returns:
- `total_cost`: The computed cost using the mean squared error formula (float).

## Vectorized Implementation:
The cost function (mean squared error) is computed using vectorized operations in NumPy:

$$J(w, b) = \frac{1}{2m} \sum_{i=1}^{m} (w \cdot x_i + b - y_i)^2 $$

The vectorized version simplifies the computation by removing the need for explicit loops. It takes advantage of NumPy's ability to perform element-wise operations on entire arrays.

