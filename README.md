# Blurry-Image-Detector

## Description:
Camera blur can be caused due to various reasons, the most common ones being out of focus and motion blur.
in case of out of focus blur, the entire image region is blurry.
in case of motion blur, it can be caused due to two reasons:
  1) Camera being in motion - this causes the entire image to have motion blur
  2) Object in motion - this causes only the object to be blurry while the rest of the image is sharp.

In order to identify the motion blur with object in motion, please look at this repo: https://github.com/Utkarsh-Deshmukh/Spatially-Varying-Blur-Detection-python

In this particular repo, we will address the "out-of-focus" blur, and motion blur(camera in motion)

## Algorithm overview:
When the image is blurry, the blurry regions undergo a high-frequency attenuation. i.e the energy in the high frequency regions goes down.
In this project, we will quantify the energy in the high-frequency content of the images and predict whether the image is blurry or sharp.

- **Step 1: Image ROI estimation:** In this step, we want to find regions in an image which we can use for further processing. We want regions which have some texture that we can analyse. Thus, we will be rejecting flat areas in the image. We use local entropy filter to get the image ROI. An example output of the image ROI estimation looks like this:
![image](https://user-images.githubusercontent.com/13918778/149880752-848b48a8-4280-4b49-8458-588850283943.png)
![image](https://user-images.githubusercontent.com/13918778/149881050-03cff68e-1114-4398-a334-d25f3d95605f.png)

- **Step 2: Feature Extraction:** In this step, we divide the image into non overlapping blocks and do a feature extraction for each block. we run this only in the region where the ROI is estimated. For each block, we compute the discrete cosine transform, and select the bottom right triangular elements from the DCT coefficients (The bottom right elements are the "high-frequency coefficients"). These elements are sorted and this sorted vector is used as the feature descriptor to train a multi-layer perceptron

- **Step 3: MultiLayer Perceptron:**  We use a multilayer perceptron with 3 fully connected layersas a classifier. The perceptron model is definer as below:
```
class MLP(nn.Module):
    def __init__(self, data_dim):
        super(MLP, self).__init__()

        self.fc1 = nn.Linear(data_dim, 64)      # Input layer
        self.fc2 = nn.Linear(64, 32)            # Hidden Layer
        self.fc3 = nn.Linear(32, 2)             # Output Layer

    def forward(self, x):
        a = self.fc1(x)                         # dot product for the first layer
        a = F.relu(a)                           # non linear activation for the first layer

        b = self.fc2(a)                         # dot product for the second layer
        b = F.relu(b)                           # non linear activation for the second layer

        c = self.fc3(b)                         # dot product for the final layer

        return c
```

## Implementation details:
 - **Dataset**: The blur image dataset from Kaggle is used for training and testing. The dataset can be found here: https://www.kaggle.com/kwentar/blur-dataset

- For Training, 1/4 of the total images were used
- For Testing, I use all the images in the dataset

## Results:

| Folder Name       | accuracy (without balancing training data)      |  accuracy (with balanced train data) |
| -------------     |:-------------:|:-------------:|
| sharp             | 72.85%  |  91.71%|
| defocussed blurred| 99.42%  | 98 %|
| motion blurred    | 95.14%  | 85.71%|

***Note: If we neglect the motion blur images, and only use the out-of focus images and the sharp images, the performance is as follows:***
| Folder Name       | accuracy      | 
| -------------     |:-------------:| 
| sharp             | 96.85% | 
| defocussed blurred| 95.14% | 


## Limitations:
- Currently, the algorithm runs at a single scale. One might consider using the approach at multi-scales to improve accuracy

## Future work:
- We can add a functionality to output a confidence score for the prediction.

## How to run the scripts:
- In order to run the feature-extraction + training of the classifier, run the file `Train_main.py`
- In order to run the prediction on images, run the file `Test_main.py`
- A pretrained classifier is provided in the folder `trained_model`.
- In the file `utils/feature_extractor.py` all the feature-extractor parameters are defined. if the parameter `blockSize_feature_extractor` or `downsamplingFactor` is changed, then the user needs to retrain the classifier (since these parameters directly affect the dimentionality of the feature vector)
