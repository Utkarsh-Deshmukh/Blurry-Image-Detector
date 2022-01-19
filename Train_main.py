import os
from utils.feature_extractor import featureExtractor
from utils.data_loader import TrainDataset
from torch.utils.data import Dataset, DataLoader
from utils.MLP import MLP
import cv2
import numpy as np
import sys
import time
import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def extract_feature_data(dataset_dir):
    img_list = os.listdir(dataset_dir)
    num_images = len(img_list)
    feature_extractor = featureExtractor()
    all_features = []
    for ind, image_name in enumerate(img_list):
        print("Feature-Extraction: %d / %d images processed.." % (ind, num_images))
        if(ind % 2 == 0):
            continue
        # Read the image
        img = cv2.imread(os.path.join(dataset_dir, image_name), 0)

        # Resize the image by the downsampling factor
        feature_extractor.resize_image(img, np.shape(img)[0], np.shape(img)[1])

        # compute the image ROI using local entropy filter
        feature_extractor.compute_roi()

        # extract the blur features using DCT transform coefficients
        extracted_features = feature_extractor.extract_feature()

        all_features.append(extracted_features)

    return(all_features)

def compile_feature_data(data, label):
    extracted_features = []
    for curr_img_data in data:
        for data_sample in curr_img_data:
            data_sample_with_label = data_sample
            data_sample_with_label.append(label)
            extracted_features.append(data_sample_with_label)
    return(extracted_features)

def compile_train_data(feature_data_blur, feature_data_sharp, feature_data_motion_blur):
    extracted_features_sharp = compile_feature_data(feature_data_sharp, label=1)
    extracted_features_blur = compile_feature_data(feature_data_blur, label=0)
    extracted_features_motion_blur = compile_feature_data(feature_data_motion_blur, label=0)
    train_data = np.concatenate((extracted_features_sharp, extracted_features_blur, extracted_features_motion_blur), axis=0)
    return(train_data)

def start_training(train_data, batch_size, num_epochs, save_model=False):
    train_data_loader = DataLoader(TrainDataset(train_data), batch_size=batch_size, shuffle=True)
    data_dim = np.shape(train_data)[1]-1
    model = MLP(data_dim).to(device)

    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(num_epochs):
        losses = []
        for batch_num, input_data in enumerate(train_data_loader):
            optimizer.zero_grad()
            x, y = input_data
            x = x.to(device).float()
            y = y.to(device)

            output = model(x)
            loss = criterion(output, y.flatten().long().to(device))
            loss.backward()
            losses.append(loss.item())

            optimizer.step()

        print('Epoch %d | Loss %6.2f' % (epoch, sum(losses) / len(losses)))
    state_dict = {'model_state': model}
    if(save_model):
        torch.save(state_dict, './trained_model/trained_model')
    return(model)

def compute_train_accuracy(trained_model, train_data):
    train_loader = DataLoader(TrainDataset(train_data), batch_size=batch_size, shuffle=True)
    total_num_samples = 0
    correct_prediction = 0
    for batch_num, input_data in enumerate(train_loader):
        x, y = input_data
        x = x.to(device).float()
        y = y.to(device)

        output = trained_model(x)
        _, predicted_label = torch.max(output, 1)

        correct_prediction += (predicted_label == y).sum().item()
        total_num_samples += output.shape[0]
    accuracy = correct_prediction / total_num_samples
    print('Train Accuracy = ')
    print(accuracy)
    return(accuracy)

def balance_data(X_Train, Y_Train):
    # Shuffle the samples
    index_arr = [i for i in range(len(Y_Train))]
    random.shuffle(index_arr)
    X_shuffled = X_Train[index_arr, :]
    Y_shuffled = Y_Train[index_arr]

    # Seperate the samples in positive and negative bin
    positive_idx = np.where(Y_shuffled == 1)
    negative_idx = np.where(Y_shuffled == 0)
    X_positive = X_shuffled[positive_idx]
    X_negative = X_shuffled[negative_idx]
    Y_positive = Y_shuffled[positive_idx]
    Y_negative = Y_shuffled[negative_idx]

    num_positive_samples = len(np.argwhere(Y_shuffled == 1))
    num_negative_samples = len(np.argwhere(Y_shuffled == 0))

    if(num_positive_samples < num_negative_samples):
        selected_X_negative = X_negative[0:num_positive_samples, :]
        selected_Y_negative = Y_negative[0:num_positive_samples]
        selected_X_positive = X_positive
        selected_Y_positive = Y_positive
    else:
        selected_X_negative = X_negative
        selected_Y_negative = Y_negative
        selected_X_positive = X_positive[0:num_positive_samples, :]
        selected_Y_positive = Y_positive[0:num_positive_samples]

    X_Train = np.concatenate((selected_X_positive, selected_X_negative), axis=0)
    Y_Train = np.concatenate((selected_Y_positive, selected_Y_negative), axis=0)

    train_data = []
    for i in range(len(Y_Train)):
        train_data.append(np.concatenate((X_Train[i], [Y_Train[i]])))

    return(train_data)

if __name__ == '__main__':
    dataset_dir = './dataset/defocused_blurred/'
    feature_data_blur = extract_feature_data(dataset_dir)

    dataset_dir = './dataset/motion_blurred/'
    feature_data_motion_blur = extract_feature_data(dataset_dir)

    dataset_dir = './dataset/sharp/'
    feature_data_sharp = extract_feature_data(dataset_dir)

    train_data = compile_train_data(feature_data_blur, feature_data_sharp, feature_data_motion_blur)

    # Balance the data
    dim = np.shape(train_data)[1]-1
    X_Train = train_data[:, 0:dim]
    Y_Train = train_data[:, -1]

    train_data = balance_data(X_Train, Y_Train)
    # Start the training
    batch_size = 1024
    num_epochs = 50

    trained_model = start_training(train_data, batch_size, num_epochs, save_model=True)

    accuracy = compute_train_accuracy(trained_model, train_data)
