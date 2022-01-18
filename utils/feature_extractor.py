import cv2
import os
import numpy as np
from skimage.filters.rank import entropy
from skimage.morphology import square

class featureExtractor(object):
    def __init__(self):
        self.blockSize_feature_extractor = 32
        self.downsamplingFactor = 2
        self.resized_image = []
        self.entropy_filt_kernel_sze = 16
        self.local_entropy_thresh = 0.6
        self.valid_img_block_thresh = 0.7
        self.roi = []
        self.__freqBands = []
        self.__dct_matrices = self.__dctmtx(self.blockSize_feature_extractor)
        self.__computeFrequencyBands()

    def __dctmtx(self, n):
        [mesh_cols, mesh_rows] = np.meshgrid(np.linspace(0, n - 1, n), np.linspace(0, n - 1, n))
        dct_matrix = np.sqrt(2 / n) * np.cos(np.pi * np.multiply((2 * mesh_cols + 1), mesh_rows) / (2 * n));
        dct_matrix[0, :] = dct_matrix[0, :] / np.sqrt(2)
        return (dct_matrix)

    def __computeFrequencyBands(self):
        current_scale = self.blockSize_feature_extractor
        matrixInds = np.zeros((current_scale, current_scale))

        for i in range(current_scale):
            matrixInds[0: max(0, int(((current_scale - 1) / 2) - i + 1)), i] = 1

        for i in range(current_scale):
            if (current_scale - ((current_scale - 1) / 2) - i) <= 0:
                matrixInds[0:current_scale - i - 1, i] = 2
            else:
                matrixInds[int(current_scale - ((current_scale - 1) / 2) - i - 1): int(current_scale - i - 1),
                i] = 2;
        matrixInds[0, 0] = 3
        self.__freqBands.append(matrixInds)

    def resize_image(self, img, rows, cols):
        self.resized_image = cv2.resize(img, (int(cols/self.downsamplingFactor), int(rows/self.downsamplingFactor)))
        rows = np.shape(self.resized_image)[0]
        cols = np.shape(self.resized_image)[1]

    def compute_roi(self):
        local_entropy = self.entropyFilt(self.resized_image)
        self.roi = 1.0 * (local_entropy > self.local_entropy_thresh*np.max(local_entropy))

    def get_single_resolution_features(self, block):
        D = self.__dct_matrices
        dct_coeff = np.abs(np.matmul(np.matmul(D, block), np.transpose(D)))
        temp = np.where(self.__freqBands[0] == 0)
        high_freq_components = dct_coeff[temp]
        high_freq_components = sorted(high_freq_components)
        return high_freq_components

    def extract_feature(self):
        extracted_features = []
        rows = np.shape(self.resized_image)[0]
        cols = np.shape(self.resized_image)[1]
        for i in range(0, rows, self.blockSize_feature_extractor):
            for j in range(0, cols, self.blockSize_feature_extractor):
                if(self.is_image_block_valid(i, j)):
                    block = self.resized_image[i : i+self.blockSize_feature_extractor, j : j+self.blockSize_feature_extractor]
                    if (np.shape(block)[0] == self.blockSize_feature_extractor) and ((np.shape(block)[1] == self.blockSize_feature_extractor)):
                        features = self.get_single_resolution_features(block)
                        extracted_features.append(features)
        return extracted_features

    def is_image_block_valid(self, i, j):
        block = self.roi[i : i+self.blockSize_feature_extractor, j : j+self.blockSize_feature_extractor]
        val = np.sum(block) / np.prod(np.shape(block))
        return val > self.valid_img_block_thresh

    def entropyFilt(self, img):
        return entropy(img, square(self.entropy_filt_kernel_sze))

    def clear_object(self):
        self.resized_image = []
        self.roi = []
        self.__freqBands = []
