# Weather Predictor Machine Learning Project

## Overview

Weather Predictor is a machine learning project developed as a part of the ECEG 478 Final Project. The primary goal of this project is to classify multi-label weather conditions and predict temperature in Celsius based on the provided input features. The project demonstrates two different models: one for weather classification and another for temperature prediction.

## Project Structure

- `predictive_model.py`: Contains the code for the temperature prediction model.
- `actual_model.py`: Contains the code for the multi-label weather classification model.
- `README.md`: This file.

## Data Description

### Weather Dataset

The dataset used for this project can be found [here](https://www.kaggle.com/datasets/swatikhedekar/python-project-on-weather-dataset).

**<span style="color:#EA401B;font-weight:bold;">Legal documents: If you want to use this code please read the LICENSE in the repository** 

### Data Attributes

- Date/Time
- Temperature in Celsius
- Dew point temperature in Celsius
- Relative humidity percentage
- Wind speed in km/h
- Visibility in km
- Pressure kPa
- Weather (Label with 50 unique weather conditions)

### Data Format

The dataset is provided in CSV format.

### Data Distribution

The dataset is imbalanced, with a higher frequency of common weather conditions like "Mainly Clear", "Mostly Cloudy", "Cloudy", and "Clear".

## Preprocessing

- **Data Cleaning**: Handle missing values and ensure data consistency.
- **Stratified Sampling**: Ensures balanced distribution of weather conditions across training, development, and testing sets.
- **Feature Engineering**: Date and Time columns are transformed into numerical features.

## Models

### Weather Classification Model

- **Architecture**: A neural network built using PyTorch.
- **Layers**:
  - Input Layer: 8 input features, 200 output features.
  - Hidden Layer: 200 input features, 100 output features.
  - Output Layer: 100 input features, number of weather conditions.
- **Activation Functions**: ReLU and Sigmoid.
- **Optimizer**: Adam and Stochastic Gradient Descent (SGD).
- **Loss Function**: Binary Cross-Entropy Loss.
- **Performance Metrics**: Accuracy, Precision, Recall, F1 Score.

### Temperature Prediction Model

- **Architecture**: A neural network built using PyTorch.
- **Layers**:
  - Input Layer: 10 input features, 500 output features.
  - Hidden Layers: 500 to 300, 300 to 100.
  - Output Layer: 100 input features, 1 output feature.
- **Activation Functions**: ReLU.
- **Optimizer**: Adam.
- **Loss Function**: Mean Squared Error (MSE).
- **Performance Metrics**: MSE, Mean Absolute Error (MAE), Root Mean Squared Error (RMSE).

## Implementation

### Weather Classification Model

```python
import torch.nn as nn

def create_model():
    model = nn.Sequential()
    model.add_module('Input Layer', nn.Linear(8, 200))
    model.add_module('ReLU Activation 1', nn.ReLU())
    model.add_module('Hidden Layer 1', nn.Linear(200, 100))
    model.add_module('Output Layer', nn.Linear(100, len(weather_conditions)))
    model.add_module('Sigmoid Activation', nn.Sigmoid())
    return model
```

## Training and Evaluation

```python
def train_and_evaluate_model(model, epochs=100):
    # Initialize lists to store performance metrics
    validation_acc_list, training_acc_list = [], []
    validation_f1_list, training_f1_list = [], []
    validation_recall_list, validation_precision_list = [], []
    cost = []

    for epoch in range(epochs):
        model.train()
        train_acc, train_precision, train_recall, train_f1 = 0.0, 0.0, 0.0, 0.0

        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            train_acc += multi_label_accuracy(outputs, labels)
            precision, recall, f1 = multi_label_f1_score(outputs, labels)
            train_precision += precision
            train_recall += recall
            train_f1 += f1
            loss.backward()
            optimizer.step()

        train_acc /= len(train_loader)
        train_precision /= len(train_loader)
        train_recall /= len(train_loader)
        train_f1 /= len(train_loader)
        training_acc_list.append(train_acc)
        training_f1_list.append(train_f1)

        model.eval()
        val_acc, val_precision, val_recall, val_f1, val_loss = 0.0, 0.0, 0.0, 0.0, 0.0

        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_acc += multi_label_accuracy(outputs, labels)
                precision, recall, f1 = multi_label_f1_score(outputs, labels)
                val_precision += precision
                val_recall += recall
                val_f1 += f1
                val_loss += loss.item()

        val_acc /= len(val_loader)
        val_precision /= len(val_loader)
        val_recall /= len(val_loader)
        val_f1 /= len(val_loader)
        validation_acc_list.append(val_acc)
        validation_f1_list.append(val_f1)
        validation_recall_list.append(val_recall)
        validation_precision_list.append(val_precision)
        val_loss /= len(val_loader)
        cost.append(val_loss)

        print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {loss.item():.4f}, Training Accuracy: {train_acc * 100:.4f}%, Training F1 Score: {train_f1:.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc * 100:.4f}%, Validation F1 Score: {val_f1:.4f}")

    return training_acc_list, validation_acc_list, training_f1_list, validation_f1_list, validation_recall_list, validation_precision_list, cost
```


## Temperature Prediction Model

```python
import torch.nn as nn

# Defining the sequential model
model = nn.Sequential()

# Add the input layer to the model. This layer has 10 input features and 500 output features.
model.add_module('Input Layer', nn.Linear(10, 500))

# Add the first ReLU (Rectified Linear Unit) activation layer.
model.add_module('ReLU Activation 1', nn.ReLU())

# Add the first hidden layer. This layer takes 500 input features (from the previous layer) and reduces to 300.
model.add_module('Hidden Layer 1', nn.Linear(500, 300))

# Add the second ReLU activation layer.
model.add_module('ReLU Activation 2', nn.ReLU())

# Add the second hidden layer. This takes 300 input features and reduces them to 100.
model.add_module('Hidden Layer 2', nn.Linear(300, 100))

# Add the third ReLU activation layer.
model.add_module('ReLU Activation 3', nn.ReLU())

# Finally, add the output layer. This layer takes 100 input features and outputs a single value.
model.add_module('Output Layer', nn.Linear(100, 1))
```
# Skills and Technologies Used
**Programming Language:** Python

Libraries:
1. <span style="color:#1B9FEA;font-weight:bold;">PyTorch: For building and training neural networks.
2. <span style="color:#1B9FEA;font-weight:bold;">Pandas: For data manipulation and preprocessing.
3. <span style="color:#1B9FEA;font-weight:bold;">NumPy: For numerical operations.
4. <span style="color:#1B9FEA;font-weight:bold;">Matplotlib: For data visualization.
5. <span style="color:#1B9FEA;font-weight:bold;">scikit-learn: For data splitting and performance metrics calculation.


Machine Learning Concepts:
1. <span style="color:#631BEA;font-weight:bold;">Multi-label Classification
2. <span style="color:#631BEA;font-weight:bold;">Regression
3. <span style="color:#631BEA;font-weight:bold;">Neural Network Design
4. <span style="color:#631BEA;font-weight:bold;">Data Preprocessing
5. <span style="color:#631BEA;font-weight:bold;">Performance Metrics (Accuracy, Precision, Recall, F1 Score, MSE, MAE, RMSE)

# Results

### Weather Classification Model
- Accuracy: 95%
- Precision, Recall, F1 Score: Low due to imbalanced dataset
### Temperature Prediction Model
- Improved Results: After correcting data splitting strategy, the model showed significant improvement in predicting temperatures with lower MSE, MAE, and RMSE.
# Conclusion
The project successfully demonstrates the implementation of machine learning models for weather classification and temperature prediction. The experience highlights the importance of data preprocessing, balanced dataset splitting, and the use of appropriate performance metrics to evaluate model performance accurately.

# Future Work
Improve data collection methods to ensure a more balanced dataset.
Explore additional features and models to enhance prediction accuracy.
Deploy the models as web services for real-time weather predictions.


