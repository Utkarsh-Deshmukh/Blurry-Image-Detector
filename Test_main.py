import os
import torch
import cv2
from utils.feature_extractor import featureExtractor
from utils.data_loader import TestDataset
from torch.utils.data import Dataset, DataLoader
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def run_testing_on_dataset(trained_model, dataset_dir, GT_blurry):
    correct_prediction_count = 0
    img_list = os.listdir(dataset_dir)
    for ind, image_name in enumerate(img_list):
        print("Blurry Image Prediction: %d / %d images processed.." % (ind, len(img_list)))

        # Read the image
        img = cv2.imread(os.path.join(dataset_dir, image_name), 0)

        prediction = is_image_blurry(trained_model, img, threshold=0.5)

        if(prediction == GT_blurry):
            correct_prediction_count += 1
    accuracy = correct_prediction_count / len(img_list)
    return(accuracy)

def is_image_blurry(trained_model, img, threshold=0.5):
    feature_extractor = featureExtractor()
    accumulator = []

    # Resize the image by the downsampling factor
    feature_extractor.resize_image(img, np.shape(img)[0], np.shape(img)[1])

    # compute the image ROI using local entropy filter
    feature_extractor.compute_roi()

    # extract the blur features using DCT transform coefficients
    extracted_features = feature_extractor.extract_feature()
    extracted_features = np.array(extracted_features)

    test_data_loader = DataLoader(TestDataset(extracted_features), batch_size=1, shuffle=False)

    # trained_model.test()
    for batch_num, input_data in enumerate(test_data_loader):
        x = input_data
        x = x.to(device).float()

        output = trained_model(x)
        _, predicted_label = torch.max(output, 1)
        accumulator.append(predicted_label.item())

    prediction= np.mean(accumulator) < threshold
    return(prediction)

if __name__ == '__main__':
    trained_model = torch.load('./trained_model/trained_model')
    trained_model = trained_model['model_state']

    dataset_dir = './dataset/defocused_blurred/'
    accuracy_blurry_images = run_testing_on_dataset(trained_model, dataset_dir, GT_blurry = True)

    dataset_dir = './dataset/sharp/'
    accuracy_sharp_images = run_testing_on_dataset(trained_model, dataset_dir, GT_blurry = False)

    dataset_dir = './dataset/motion_blurred/'
    accuracy_motion_blur_images = run_testing_on_dataset(trained_model, dataset_dir, GT_blurry=True)

    print("========================================")
    print('Test accuracy on blurry forlder = ')
    print(accuracy_blurry_images)

    print('Test accuracy on sharp forlder = ')
    print(accuracy_sharp_images)

    print('Test accuracy on motion blur forlder = ')
    print(accuracy_motion_blur_images)
