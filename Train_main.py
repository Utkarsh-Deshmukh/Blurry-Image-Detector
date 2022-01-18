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

if __name__ == '__main__':
    dataset_dir = './dataset/defocused_blurred/'
    feature_data_blur = extract_feature_data(dataset_dir)

    dataset_dir = './dataset/motion_blurred/'
    feature_data_motion_blur = extract_feature_data(dataset_dir)

    dataset_dir = './dataset/sharp/'
    feature_data_sharp = extract_feature_data(dataset_dir)

    train_data = compile_train_data(feature_data_blur, feature_data_sharp, feature_data_motion_blur)

    # Start the training
    batch_size = 1024
    num_epochs = 50

    trained_model = start_training(train_data, batch_size, num_epochs, save_model=True)

    accuracy = compute_train_accuracy(trained_model, train_data)
