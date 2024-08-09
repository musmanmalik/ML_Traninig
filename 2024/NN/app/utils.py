import torch
import torch.nn as nn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms


def prepare_dataset():
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42, shuffle=True)

    X_tr_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_tr_tensor = torch.tensor(y_train, dtype=torch.long)
    X_test = torch.tensor(X_test, dtype=torch.float32)

    return X_tr_tensor, y_tr_tensor, X_test, y_test


class S_N_Network(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.l1 = nn.Linear(input_size, 64)  
        self.act1 = nn.ReLU()       
        self.l2 = nn.Linear(64, 16) 
        self.drop = nn.Dropout(0.2) 
        self.act2 = nn.ReLU()       
        self.l3 = nn.Linear(16, num_classes)  
    def forward(self, x):
        x = self.l1(x)
        x = self.act1(x)
        x = self.l2(x)
        x = self.drop(x)
        x = self.act2(x)
        x = self.l3(x)
        return x


def fit_iris(model, X_tr_tensor, y_tr_tensor, lr=0.002):
    epochs = 400
    loss_arr = []
    loss_fn = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        ypred = model(X_tr_tensor)
        loss = loss_fn(ypred, y_tr_tensor)
        loss_arr.append(loss.item())
        loss.backward()
        optim.step()
        optim.zero_grad()
    return loss_arr

    
def test(model, X_test_tensor, y_test):
    # X_ts_tensor = torch.tensor(X_test, dtype=torch.float32).to('cuda:0' if torch.cuda.is_available() else "cpu")
    ytest_pred = model(X_test_tensor)
    newytest = torch.argmax(ytest_pred, dim=1)
    accuracy = accuracy_score(newytest.cpu(), y_test)
    c_matrix = confusion_matrix(newytest.cpu(), y_test)
    return accuracy, c_matrix

def predict_iris(model, X_ts_tensor):
    # X_ts_tensor = torch.tensor(X_test, dtype=torch.float32).to('cuda:0' if torch.cuda.is_available() else "cpu")
    ytest_pred = model(X_ts_tensor)
    newytest = torch.argmax(ytest_pred, dim=1)
    return newytest.cpu()
    
    
def save_plot(path, data, title, x_label=None, y_label=None):
    print(path)
    plt.figure(figsize=(6, 4))
    plt.plot(data)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig(path, format='png')
    plt.close()
    print("Image saved")
    
    
    
def fit_minist(model, train_loader, lr=0.002):
    epochs = 8
    loss_arr = []
    loss_fn = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    # Train Network
    for epoch in range(epochs):  
    # 1 epoch => Network has seen all the images in the dataset
        
        print(f'Epoch: {epoch}')

        for batch_idx, (data, targets) in enumerate(train_loader):

            data = data.to('cuda:0' if torch.cuda.is_available() else "cpu")
            targets = targets.to('cuda:0' if torch.cuda.is_available() else "cpu")

            # print(data.shape)  # => [64 , 1, 28, 28] => 64 : num_images, 1 -> num_channels, (28,28): (height, width)
            data = data.reshape(data.shape[0], -1)  # Flatten
            if epoch == 0 and batch_idx == 0:
                print(data.shape)

            # forward
            scores = model(data)

            loss = loss_fn(scores, targets)
            loss_arr.append(loss.item())

            # backward
            optim.zero_grad()  # set all gradients to zero for each batch
            loss.backward()

            # gradient descent
            optim.step()
    return loss_arr

def test_minist_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():  # dont compute gradients
        for x, y in loader:
            x = x.to('cuda:0' if torch.cuda.is_available() else "cpu")
            y = y.to('cuda:0' if torch.cuda.is_available() else "cpu")
            x = x.reshape(x.shape[0],-1)

            scores = model(x)
            _, predictions = scores.max(1)

            # print(predictions)

            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        # print(f'Got {num_correct}/{num_samples} with accuracy {float(num_correct)/float(num_samples)*100: .2f}')

    model.train()
    return float(num_correct)/float(num_samples)



# Define the transformations
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def predict_image(image, model):
    try:
        image = transform(image)
        image = image.view(-1, 784)  # Flatten the image
        image = image.to('cuda:0' if torch.cuda.is_available() else "cpu")
        
        with torch.no_grad():
            output = model(image)
            _, predicted = torch.max(output, 1)
        
        # print(predicted.item())
        return predicted.item()
    except Exception as e:
        print(f"Error in prediction: {e}")
        return None
