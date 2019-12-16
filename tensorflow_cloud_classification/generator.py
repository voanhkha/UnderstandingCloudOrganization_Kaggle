from keras.utils import Sequence
import os
from copy import deepcopy
import random
import numpy as np
import cv2



class DataGenenerator(Sequence):
    def __init__(self, images_list=None, folder_imgs='../../dados/train_images',
                 img_2_ohe_vector=None,batch_size=32, shuffle=True, augmentation=None,
                 resized_height=260, resized_width=260, num_channels=3):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augmentation = augmentation
        if images_list is None:
            self.images_list = os.listdir(folder_imgs)
        else:
            self.images_list = deepcopy(images_list)
        self.folder_imgs = folder_imgs
        self.len = len(self.images_list) // self.batch_size
        self.resized_height = resized_height
        self.resized_width = resized_width
        self.num_channels = num_channels
        self.num_classes = 4
        self.is_test = not 'train' in folder_imgs

        self.img_2_ohe_vector = img_2_ohe_vector
        if not shuffle and not self.is_test:
            self.labels = [self.img_2_ohe_vector[img] for img in self.images_list[:self.len * self.batch_size]]

    def __len__(self):
        return self.len

    def on_epoch_start(self):
        if self.shuffle:
            random.shuffle(self.images_list)

    def __getitem__(self, idx):
        current_batch = self.images_list[idx * self.batch_size: (idx + 1) * self.batch_size]
        X = np.empty((self.batch_size, self.resized_height, self.resized_width, self.num_channels))
        y = np.empty((self.batch_size, self.num_classes))

        for i, image_name in enumerate(current_batch):
            path = os.path.join(self.folder_imgs, image_name)
            img = cv2.resize(cv2.imread(path), (self.resized_height, self.resized_width)).astype(np.float32)
            if not self.augmentation is None:
                augmented = self.augmentation(image=img)
                img = augmented['image']
            X[i, :, :, :] = img / 255.0
            if not self.is_test:
                y[i, :] = self.img_2_ohe_vector[image_name]
        return X, y

    def get_labels(self):
        if self.shuffle:
            images_current = self.images_list[:self.len * self.batch_size]
            labels = [self.img_2_ohe_vector[img] for img in images_current]
        else:
            labels = self.labels
        return np.array(labels)
