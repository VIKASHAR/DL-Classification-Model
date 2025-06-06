# Developing a Neural Network Classification Model

## AIM
To develop a neural network classification model for the given dataset.

## THEORY
The Iris dataset consists of 150 samples from three species of iris flowers (Iris setosa, Iris versicolor, and Iris virginica). Each sample has four features: sepal length, sepal width, petal length, and petal width. The goal is to build a neural network model that can classify a given iris flower into one of these three species based on the provided features.

## DESIGN STEPS
### STEP 1: 

Load the Iris dataset using a suitable library.

### STEP 2: 

Preprocess the data by handling missing values and normalizing features.

### STEP 3: 

Split the dataset into training and testing sets.

### STEP 4: 

Train a classification model using the training data

### STEP 5: 

Evaluate the model on the test data and calculate accuracy

### STEP 6: 

Display the test accuracy, confusion matrix, and classification report.

## PROGRAM

### Name: Vikash A R

### Register Number: 212222040179

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
     

class Model(nn.Module):
    def __init__(self, in_features=4, h1=8, h2=9, out_features=3):
        super().__init__()
        self.fc1 = nn.Linear(in_features,h1)    # input layer
        self.fc2 = nn.Linear(h1, h2)            # hidden layer
        self.out = nn.Linear(h2, out_features)  # output layer

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x

torch.manual_seed(32)
model = Model()
     
df = pd.read_csv('/content/iris.csv')
df.head()
     
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10,7))
fig.tight_layout()

for i, ax in enumerate(axes.flat):
    for j in range(3):
        x = df.columns[plots[i][0]]
        y = df.columns[plots[i][1]]
        ax.scatter(df[df['target']==j][x], df[df['target']==j][y], color=colors[j])
        ax.set(xlabel=x, ylabel=y)

fig.legend(labels=labels, loc=3, bbox_to_anchor=(1.0,0.85))
plt.show()
     
X = df.drop('target',axis=1).values
y = df['target'].values

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=33)

X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
# y_train = F.one_hot(torch.LongTensor(y_train))  # not needed with Cross Entropy Loss
# y_test = F.one_hot(torch.LongTensor(y_test))
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)
     
trainloader = DataLoader(X_train, batch_size=60, shuffle=True)
testloader = DataLoader(X_test, batch_size=60, shuffle=False)
     
torch.manual_seed(4)
model = Model()
     
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

epochs = 100
losses = []

for i in range(epochs):
    i+=1
    y_pred = model.forward(X_train)
    loss = criterion(y_pred, y_train)
    losses.append(loss)

    # a neat trick to save screen space:
    if i%10 == 1:
        print(f'epoch: {i:2}  loss: {loss.item():10.8f}')

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

#plt.plot(range(epochs), losses)
#plt.ylabel('Loss')
#plt.xlabel('epoch');
plt.plot(range(epochs), [loss.item() for loss in losses])
plt.ylabel('Loss')
plt.xlabel('epoch');
torch.save(model.state_dict(),'Sri Sai Priya S.pt')
```

### OUTPUT

![image](https://github.com/user-attachments/assets/701227d7-ceb4-4446-afa3-19ddd6790375)

![image](https://github.com/user-attachments/assets/10bf5c2c-7ab8-454d-91cf-8e3fb073384d)

![image](https://github.com/user-attachments/assets/63682127-7bab-422e-a2a5-db589e82ba18)

![image](https://github.com/user-attachments/assets/f82d300d-1772-458b-b6df-1da3d5170a15)

## RESULT

Thus, a neural network classification model was successfully developed and trained using PyTorch.
