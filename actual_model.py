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
import numpy as np
from sklearn.utils.class_weight import compute_class_weight


def multi_label_f1_score(predictions, targets, threshold=0.3):
    predictions = (predictions > threshold).int()
    # print(f"predictions: {predictions.tolist()}")
    targets = targets.int()
    # print(f"targets:     {targets.tolist()}")

    true_positives = (predictions & targets).sum(dim=0).float()
    # check if we calculate the true positives correctly because true positives and targets do not make sense
    false_positives = (predictions & ~targets).sum(dim=0).float()
    false_negatives = (~predictions & targets).sum(dim=0).float()

    precision = true_positives / (true_positives + false_positives + 1e-8)
    recall = true_positives / (true_positives + false_negatives + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return precision.mean().item(), recall.mean().item(), f1.mean().item()



def multi_label_accuracy(predictions, targets, threshold=0.4):
    """
    Compute the accuracy for multi-label classification
    :param predictions: The model predictions (sigmoid outputs)
    :param targets: The ground truth labels
    :param threshold: The threshold for converting sigmoid outputs to binary outputs
    :return: Accuracy score
    """
    # Apply threshold to predictions to get binary outputs
    predictions = predictions > threshold


    # Convert to integer type
    predictions = predictions.int()
    targets = targets.int()

    # Compute accuracy
    correct_predictions = (predictions == targets).float()  # Convert to float for division

    accuracy = correct_predictions.sum() / correct_predictions.numel()
    return accuracy.item()


def calculate_accuracy(model, data_loader):
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            outputs = model(inputs)
            predictions = outputs > 0.5  # Apply threshold
            all_predictions.extend(predictions.numpy().flatten())
            all_labels.extend(labels.numpy().flatten())

    # Calculate accuracy
    accuracy = accuracy_score(all_labels, all_predictions)
    return accuracy



weather_conditions = ['Clear', 'Cloudy', 'Drizzle', 'Fog', 'Freezing Drizzle', 'Freezing Fog', 'Freezing Rain',
                          'Haze', 'Mainly Clear', 'Moderate Snow', 'Mostly Cloudy', 'Rain', 'Rain Showers', 'Snow',
                          'Snow Showers', 'Thunderstorms', 'Blowing Snow', 'Ice Pellets', 'Heavy Rain Showers',
                          'Moderate Rain', 'Moderate Rain Showers', 'Snow Grains', 'Snow Pellets']
def encode_weather(file_path, condition=False):
    # global data, label_df
    data = pd.read_csv(file_path)
    # List of weather conditions for one-hot encoding

    # Initialize MultiLabelBinarizer
    mlb = MultiLabelBinarizer(classes=weather_conditions)
    # print(f"mlb {mlb}")
    # Splitting the 'Weather' column into a list of conditions
    data['Weather'] = data['Weather'].str.split(',')

    # One-hot encode the weather conditions
    one_hot_labels = mlb.fit_transform(data['Weather'])
    if condition:
        ones_list = []
        for i in range(len(one_hot_labels)):
            for j in range(len(one_hot_labels[i])):
                if one_hot_labels[i][j] == 1:
                    ones_list.append(j)

            print(f"Current weather condition -> {data['Weather'][i]}, encoded labels -> {one_hot_labels[i]} | checking -> {ones_list}, {[weather_conditions[element] for element in ones_list]}")
            ones_list.clear()
    # print(f"one_hot_labels: \n{one_hot_labels}")
    # Convert the one-hot encoded labels to a DataFrame
    label_df = pd.DataFrame(one_hot_labels, columns=mlb.classes_)
    # Extract the feature columns (first 8 columns)
    # feature_columns = data.columns[:8]
    # features = data[feature_columns]
    # print("data")
    # print(data)
    # print("label_df")
    # print(len(label_df.values))     # 6122 & 1364
    return data, label_df


def standardize_date_time(data):
    scaler1 = StandardScaler()
    X = data.iloc[:, :8]
    X['Date'] = pd.to_datetime(X['Date']).map(pd.Timestamp.timestamp)
    X['Time'] = X['Time'].apply(lambda x: int(x.split(':')[0]) * 60 + int(x.split(':')[1]))
    # Convert labels to numpy array
    X_scaled = scaler1.fit_transform(X)
    return X_scaled

def create_model():
    # global model
    model = nn.Sequential()
    model.add_module('Input Layer', nn.Linear(8, 200))
    # model.add_module('Input Layer', nn.Linear(8, len(weather_conditions)))
    # model.add_module('ReLU Activation 1', nn.ReLU())
    # model.add_module('Hidden Layer 1', nn.Linear(500, 300))
    model.add_module('ReLU Activation 2', nn.ReLU())
    model.add_module('Hidden Layer 2', nn.Linear(200, 100))
    # model.add_module('ReLU Activation 2', nn.ReLU())
    model.add_module('Output Layer', nn.Linear(100, len(weather_conditions)))
    model.add_module('Sigmoid Activation', nn.Sigmoid())
    return model



def train_and_evaluate_model(model, epochs=100):
    validation_acc_list = []
    training_acc_list = []
    validation_f1_list = []
    training_f1_list = []
    validation_recall_list = []
    validation_precision_list = []
    cost = []

    for epoch in range(epochs):
        model.train()
        train_acc = 0.0
        train_precision = 0.0
        train_recall = 0.0
        train_f1 = 0.0

        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            train_acc += multi_label_accuracy(outputs, labels)

            # Calculate precision, recall, and F1 score
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
        val_acc = 0.0
        val_precision = 0.0
        val_recall = 0.0
        val_f1 = 0.0
        val_loss = 0.0


        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_acc += multi_label_accuracy(outputs, labels)

                # Calculate precision, recall, and F1 score
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
        cost.append(loss.item())

        print(
            f"Epoch {epoch + 1}/{epochs}, Training Loss: {loss.item():.4f}, Training Accuracy: {train_acc * 100:.4f}%, Training F1 Score: {train_f1:.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc * 100:.4f}%, Validation F1 Score: {val_f1:.4f}")

    return training_acc_list, validation_acc_list, training_f1_list, validation_f1_list, validation_recall_list, validation_precision_list, cost


if __name__ == '__main__':

    file_path = 'DataSets/training_data.csv'
    print(f"Encoding weather started for {os.path.basename(file_path)} file")
    data, label_df = encode_weather(file_path, condition=False)
    print(f"Encoding weather completed for {os.path.basename(file_path)} file")

    print("Standardizing X_train Started")
    X_train = standardize_date_time(data)
    print(X_train)
    print("Standardizing X_train Completed")

    y = label_df.values
    print(y)
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    # y_train_tensor = torch.tensor(y, dtype=torch.int64)
    y_train_tensor = torch.tensor(y, dtype=torch.float32)



    # testing_file_path = 'DataSets/test_data.csv'
    testing_file_path = 'DataSets/development_data.csv'
    print(f"Encoding weather started for {os.path.basename(testing_file_path)} file")
    test_data, label_df_test = encode_weather(testing_file_path, condition=False)
    print(f"Encoding weather completed for {os.path.basename(testing_file_path)} file")
    # print(f"label_df_test\n {label_df_test.to_string()}")
    print("Standardizing X_test Started")
    X_test = standardize_date_time(test_data)
    print("Standardizing X_test Completed")

    y_test = label_df_test.values

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    # y_test_tensor = torch.tensor(y_test, dtype=torch.int64)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)



    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    # Creating DataLoaders
    print("Creating DataLoaders")
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True)






    model = create_model()
    print("model is successfully created")

    # Loss function and optimizer
    criterion = nn.BCELoss()  # Binary Cross-Entropy Loss for multi-label classification lr = 0.001
    # criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    # optimizer = optim.SGD(model.parameters(), lr=0.001)

    epochs = 200  # Number of epochs
    training_acc_list, validation_acc_list, training_f1_list, validation_f1_list, validation_recall_list, validation_precision_list, cost = train_and_evaluate_model(model, epochs=epochs)


    plt.plot([epoch for epoch in range(epochs)], training_acc_list, label='Training Accuracy')
    plt.plot([epoch for epoch in range(epochs)], validation_acc_list, label='Testing Accuracy')
    plt.plot([epoch for epoch in range(epochs)], cost, label='Cost')
    # plt.plot([epoch for epoch in range(epochs)], validation_f1_list, label='Testing F1 Score')

    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(['Training Accuracy', 'Testing Accuracy', 'Cost', 'Testing F1 Score'])
    plt.xlim([-5, epochs])
    plt.ylim([0, 1])
    plt.show()
    plt.plot([epoch for epoch in range(epochs)], validation_recall_list, label='Testing Recall')
    plt.plot([epoch for epoch in range(epochs)], validation_precision_list, label='Testing Precision')
    plt.legend(['Testing Recall', 'Testing Precision'])
    plt.xlabel('Epochs')
    plt.ylabel('Score')

    plt.show()



