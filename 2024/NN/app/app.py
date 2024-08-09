
import os
import torch
import matplotlib.pyplot as plt
import joblib
import numpy as np
from PIL import Image 


from torch.utils.data import DataLoader  # easier dataset management, helps create mini batches
import torchvision.datasets as datasets  # standard datasets
import torchvision.transforms as transforms

from flask import Flask, render_template, request, jsonify
from utils import S_N_Network, prepare_dataset, fit_iris, test, predict_iris
from utils import fit_minist, test_minist_accuracy, predict_image

app = Flask(__name__)

#IRIS Model and releated stuff at global level
X_tr_tensor, y_tr_tensor, X_test_tensor, y_test = prepare_dataset()
input_parameters = 4
output_labels = 3
iris_model = S_N_Network(input_parameters,output_labels).to('cuda:0' if torch.cuda.is_available() else "cpu")
iris_model.to('cuda:0' if torch.cuda.is_available() else "cpu")  # Move model to GPU
X_tr_tensor = X_tr_tensor.to('cuda:0' if torch.cuda.is_available() else "cpu")  
y_tr_tensor = y_tr_tensor.to('cuda:0' if torch.cuda.is_available() else "cpu")
X_test_tensor = X_test_tensor.to('cuda:0' if torch.cuda.is_available() else "cpu")
loss_arr = fit_iris(iris_model, X_tr_tensor, y_tr_tensor)
# save_plot(plot_path, loss_arr, 'IRIS LOSS PLOT', x_label=None, y_label=None)

#minist data and related stuff

batch_size = 64

# Load Data
train_dataset = datasets.MNIST(root='dataset/', train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = datasets.MNIST(root='dataset/', train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
mininst_input_parameters = 784
mininst_output_labels = 10
minist_model = S_N_Network(mininst_input_parameters,mininst_output_labels).to('cuda:0' if torch.cuda.is_available() else "cpu")
loss_arr_minist = fit_minist(minist_model, train_loader)



@app.route('/')
def index():
    return render_template('index.html')


# #iris_loss
# @app.route('/iris_loss')
# def get_iris_loss_plot():
    



@app.route('/action')
def test_model_iris():
    # print(loss_arr)
    accuracy, c_matrix = test(iris_model, X_test_tensor, y_test)
    # accuracy = test_minist_accuracy(test_loader, minist_model)
    
    return "Model Accuracy of Test Data: "+str(accuracy)


@app.route('/minist_test_accuracy')
def minist_test_accuracy():
    # print(loss_arr)
    accuracy = test_minist_accuracy(test_loader, minist_model)
    
    return "Model Accuracy of Test Data: "+str(accuracy)



@app.route('/predict_iris', methods=['POST'])
def predict():
    try:
        # Parse the input features from the request
        data = request.json
        # print(data['features'])
        features = np.array(data['features']).reshape(1, -1)
        # print(features)
        
        # Make prediction
        features = torch.tensor(features, dtype=torch.float32).to('cuda:0' if torch.cuda.is_available() else "cpu")
        prediction = predict_iris(iris_model, features)
        # print(prediction)
        # Map prediction to class names
        class_names = ['setosa', 'versicolor', 'virginica']
        result = class_names[prediction]
        print(result)
        return jsonify({'prediction': result})
    except Exception as e:
        return jsonify({'error': str(e)})
    
    
    
@app.route('/predict_minist', methods=['POST'])
def predict_minist():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        # Load and preprocess the image
        image = Image.open(file).convert('RGB')  # Convert image to RGB
        predicted_class = predict_image(image, minist_model)
        return jsonify({'prediction': predicted_class})

    return jsonify({'error': 'File processing error'}), 500



if __name__ == '__main__':
    app.run(debug=True)
