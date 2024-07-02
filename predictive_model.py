from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import datetime

def prepare_data_modified(df):
    # Convert Date and Time into a single datetime object
    df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])

    # Extract year, month, day, and hour as separate features
    df['Year'] = df['DateTime'].dt.year
    df['Month'] = df['DateTime'].dt.month
    df['Day'] = df['DateTime'].dt.day
    df['Hour'] = df['DateTime'].dt.hour

    # Drop the original Date, Time, and DateTime columns
    df.drop(['Date', 'Time', 'DateTime'], axis=1, inplace=True)

    # Encode the categorical 'Weather' column
    label_encoder = LabelEncoder()
    df['Weather'] = label_encoder.fit_transform(df['Weather'])

    # Extract features and target variable
    X = df.drop('Temp_C', axis=1)
    y = df['Temp_C']

    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print(f"X_scaled {X_scaled}")
    print(f"y {y}")

    return X_scaled, y


# Prepare the training and development data with the modified function
training_data_path = 'DataSets2/training_data.csv'
# development_data_path = 'DataSets/development_data.csv'
development_data_path = 'DataSets2/test_data.csv'

training_data = pd.read_csv(training_data_path)
development_data = pd.read_csv(development_data_path)
X_train, y_train = prepare_data_modified(training_data)
X_dev, y_dev = prepare_data_modified(development_data)
print("y_train")
print(y_train)


load_model = False
if load_model:
    print('Loading model...')
    model = torch.load('temperature_prediction_model.pt')
else:
    model = nn.Sequential()
    model.add_module('Input Layer', nn.Linear(10, 500))
    # model.add_module('Input Layer', nn.Linear(8, len(weather_conditions)))
    model.add_module('ReLU Activation 1', nn.ReLU())
    model.add_module('Hidden Layer 1', nn.Linear(500, 300))
    model.add_module('ReLU Activation 2', nn.ReLU())
    model.add_module('Hidden Layer 2', nn.Linear(300, 100))
    model.add_module('ReLU Activation 2', nn.ReLU())
    model.add_module('Output Layer', nn.Linear(100, 1))



# Convert datasets to tensors
X_train_tensor = torch.tensor(X_train).float()
y_train_tensor = torch.tensor(y_train.values).float().unsqueeze(1)  # Reshape for compatibility
X_dev_tensor = torch.tensor(X_dev).float()
y_dev_tensor = torch.tensor(y_dev.values).float().unsqueeze(1)

# Create data loaders
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Define the loss function and the optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.00001)
# optimizer = optim.SGD(model.parameters(), lr=0.00001)
training_cost = []
development_cost = []
train_mae, train_rmse = [], []
dev_mae, dev_rmse = [], []
if load_model == False:
    # Training loop
    num_epochs = 60  # Define the number of epochs

    for epoch in range(num_epochs):
        model.train()
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            train_mae_epoch = mean_absolute_error(targets.numpy(), outputs.detach().numpy())
            train_rmse_epoch = np.sqrt(mean_squared_error(targets.numpy(), outputs.detach().numpy()))
            loss.backward()
            optimizer.step()

        # Validation step
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            dev_outputs = model(X_dev_tensor)
            dev_loss = criterion(dev_outputs, y_dev_tensor)
            dev_mae_epoch = mean_absolute_error(y_dev, dev_outputs.numpy())
            dev_rmse_epoch = np.sqrt(mean_squared_error(y_dev, dev_outputs.numpy()))
            val_loss += dev_loss.item()
        development_cost.append(dev_loss.item())
        training_cost.append(loss.item())
        train_mae.append(train_mae_epoch)
        train_rmse.append(train_rmse_epoch)
        dev_mae.append(dev_mae_epoch)
        dev_rmse.append(dev_rmse_epoch)
        print(f'Epoch {epoch+1}, Loss: {loss.item()}, Dev Loss: {dev_loss.item()}')

    # Save the model if needed
    # torch.save(model, 'temperature_prediction_model.pt')

# Make predictions on the development set
model.eval()
with torch.no_grad():
    predictions = model(X_dev_tensor).squeeze()

# Convert predictions back to a numpy array
predictions_np = predictions.numpy()

# Calculate evaluation metrics
mse = mean_squared_error(y_dev, predictions_np)
mae = mean_absolute_error(y_dev, predictions_np)
rmse = np.sqrt(mse)

print(f'Mean Squared Error: {mse}')
print(f'Mean Absolute Error: {mae}')
print(f'Root Mean Squared Error: {rmse}')


y_dev_np = y_dev.to_numpy()
y_train_np = y_train.to_numpy()

# Plotting predictions vs actuals with different colors
num_list = [element for element in range(len(predictions_np))]
num_list_train = [element for element in range(len(y_train_np))]


print(list(y_dev_np))

print(list(predictions_np))

fig, axs = plt.subplots(2, 2, figsize=(12, 10))

axs[0, 0].scatter(y_dev_np, predictions_np, alpha=0.5)
axs[0, 0].plot([-20, 50], [-20, 50], '-r', label='y=x')
axs[0, 0].set_xlim(-20, 50)
axs[0, 0].set_ylim(-20, 50)
axs[0, 0].set_xlabel('Actual Temperatures')
axs[0, 0].set_ylabel('Predicted Temperatures')

if development_data_path == 'DataSets/test_data.csv':
    axs[0, 0].set_title('Actual vs Predicted Temperatures Test Set')
else:
    axs[0, 0].set_title('Actual vs Predicted Temperatures Development Set')

axs[0, 1].plot(training_cost, label='Training Cost')
if development_data_path == 'DataSets/test_data.csv':
    axs[0, 1].plot(development_cost, label='Testing Cost')
else:
    axs[0, 1].plot(development_cost, label='Development Cost')
axs[0, 1].set_xlabel('Number of Epochs')
axs[0, 1].set_ylabel('Mean Squared Error')
axs[0, 1].set_title("Mean Squared Error vs Number of Epochs")
axs[0, 1].legend()

axs[1, 0].plot(train_mae, label='Training MAE')
if development_data_path == 'DataSets/test_data.csv':
    axs[1, 0].plot(dev_mae, label='Testing MAE')
else:
    axs[1, 0].plot(dev_mae, label='Development MAE')
axs[1, 0].set_xlabel('Number of Epochs')
axs[1, 0].set_ylabel('Mean Absolute Error')
axs[1, 0].set_title("Mean Absolute Error vs Number of Epochs")
axs[1, 0].legend()

axs[1, 1].plot(train_rmse, label='Training RMSE')
if development_data_path == 'DataSets/test_data.csv':
    axs[1, 1].plot(dev_rmse, label='Testing RMSE')
else:
    axs[1, 1].plot(dev_rmse, label='Development RMSE')
axs[1, 1].set_xlabel('Number of Epochs')
axs[1, 1].set_ylabel('Root Mean Squared Error')
axs[1, 1].set_title("Root Mean Squared Error vs Number of Epochs")
axs[1, 1].legend()

plt.tight_layout()
plt.show()



