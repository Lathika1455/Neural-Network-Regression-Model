# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

This code builds and trains a feedforward neural network in PyTorch for a regression task. The model takes a single input feature, passes it through two hidden layers with ReLU activation, and predicts one continuous output. It uses MSE loss and RMSProp optimizer to minimize the error between predictions and actual values over training epochs.

## Neural Network Model

<img width="1235" height="779" alt="image" src="https://github.com/user-attachments/assets/8566accd-bec2-48e8-b303-7a669c27837b" />


## DESIGN STEPS

### STEP 1:

Create your dataset in a Google sheet with one numeric input and one numeric output.

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

### STEP 8:

Use the trained model to predict for a new input value.


## PROGRAM
### Name: LATHIKA SREE R
### Register Number: 212224040169
```
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

dataset1 = pd.read_csv('sample.csv')
X = dataset1[['Input']].values
y = dataset1[['Output']].values

dataset1.head(5)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=33)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# Name: Lathika Sree R
# Register Number: 212224040169
class NeuralNet(nn.Module):
  def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1,8)
        self.fc2 = nn.Linear(8,10)
        self.fc3 = nn.Linear(10,1)
        self.relu = nn.ReLU()
        self.history = {'loss': []}

  def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize the Model, Loss Function, and Optimizer
Lathika = NeuralNet()
criterion = nn.MSELoss()
optimizer = optim.RMSprop(Lathika.parameters(), lr=0.001)

# Name: LATHIKA SREE R
# Register Number: 212224040169
def train_model(Lathika, X_train, y_train, criterion, optimizer, epochs=2000):
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = criterion(Lathika(X_train), y_train)
        loss.backward()
        optimizer.step()


        # Append loss inside the loop
        Lathika.history['loss'].append(loss.item())

        if epoch % 200 == 0:
            print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}')


train_model(Lathika, X_train_tensor, y_train_tensor, criterion, optimizer)

with torch.no_grad():
    test_loss = criterion(Lathika(X_test_tensor), y_test_tensor)
    print(f'Test Loss: {test_loss.item():.6f}')

loss_df = pd.DataFrame(Lathika.history)

import matplotlib.pyplot as plt
print("\nName: Lathika sree R")
print("Register Number: 212224040169")
loss_df.plot()
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss during Training")
plt.show()

X_n1_1 = torch.tensor([[9]], dtype=torch.float32)
prediction = Lathika(torch.tensor(scaler.transform(X_n1_1), dtype=torch.float32)).item()
print("Name: Lathika Sree R")
print("Register Number:212224040169")
print(f'Prediction: {prediction}')
```
## Dataset Information

<img width="173" height="229" alt="image" src="https://github.com/user-attachments/assets/33b1ccb1-3773-45d5-b391-3483adf90f4f" />


## OUTPUT

<img width="459" height="309" alt="image" src="https://github.com/user-attachments/assets/90ce9af8-02be-4aaf-90b0-b46fb0795e16" />


### Training Loss Vs Iteration Plot

<img width="721" height="575" alt="image" src="https://github.com/user-attachments/assets/1241c228-e95e-4a4b-a310-7f4bddaad769" />


### New Sample Data Prediction

<img width="304" height="71" alt="image" src="https://github.com/user-attachments/assets/e6ca9035-fd60-45ad-bb24-3b0241b25654" />


## RESULT

Thus, a neural network regression model was successfully developed and trained using PyTorch.
