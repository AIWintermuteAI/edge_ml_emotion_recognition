import numpy as np
import cv2
import os 
import imgaug as ia
import logging
from imgaug import augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

def _create_augment_pipeline():

    sometimes = lambda aug: iaa.Sometimes(0.1, aug)

    aug_pipe = iaa.Sequential(
        [
            iaa.Fliplr(0.5), 
            #iaa.Flipud(0.2), 
            iaa.Affine(translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)}),
            iaa.OneOf([iaa.Affine(scale=(0.8, 1.2)),
                        iaa.Affine(rotate=(-10, 10)),
                        iaa.Affine(shear=(-10, 10))]),

                        sometimes(iaa.OneOf([
                               iaa.GaussianBlur((0, 3.0)),
                               iaa.AverageBlur(k=(2, 7)),
                               iaa.MedianBlur(k=(3, 11)),
                           ])),
                           sometimes(iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5))),
                           sometimes(iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5)),
                           sometimes(iaa.OneOf([
                               iaa.Dropout((0.01, 0.1), per_channel=0.5),
                               iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
                           ])),
                           sometimes(iaa.Add((-10, 10), per_channel=0.5)),  
                           sometimes(iaa.Multiply((0.5, 1.5), per_channel=0.5)), 
                           sometimes(iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5)) 
        ],
        random_order=True
    )

    return aug_pipe

def process_image_classification(image, desired_w = None, desired_h = None, aug_pipe = None):
    
    # resize the image to standard size
    if (desired_w and desired_h) or aug_pipe:

        if (desired_w and desired_h):
            # Rescale image
            image = ia.imresize_single_image(image, (desired_w, desired_h))

        if aug_pipe:
            image = aug_pipe(image=image)
        
    return image


class DataGenerator():
    def __init__(self, X_train, y_train, batch_size=32, img_size = 48, prefix='appa-real/imgs/', shuffle=True, augment=None):
        self.X_train = X_train
        self.y_train = y_train
        self.batch_size = batch_size
        self.img_size = img_size
        self.prefix = prefix
        self.class_num = y_train.shape[1]

        self.shuffle = shuffle
        self.sample_num = len(X_train)
        self.augment = augment
        if self.augment:
            logging.info("Using augmentation for {self.prefix}")
            self.aug_pipe = _create_augment_pipeline()

    def __call__(self):
        while True:
            indexes = self.__get_exploration_order()
            itr_num = int(len(indexes) // (self.batch_size * 2))

            for i in range(itr_num):
                batch_ids = indexes[i * self.batch_size * 2:(i + 1) * self.batch_size * 2]
                X, y = self.__data_generation(batch_ids)

                yield X, y

    def __get_exploration_order(self):
        indexes = np.arange(self.sample_num)

        if self.shuffle:
            np.random.shuffle(indexes)

        return indexes

    def __data_generation(self, batch_ids):
        X = np.zeros(shape=(self.batch_size, self.img_size, self.img_size, 3))
        y = np.zeros(shape=(self.batch_size, self.class_num))

        for i in range(self.batch_size):
            img = cv2.imread(self.prefix + self.X_train[batch_ids[i]], 1)

            try:
                if self.augment:
                    img = process_image_classification(img, self.img_size, self.img_size, self.aug_pipe)
            except Exception as e:
                print(self.prefix + self.X_train[batch_ids[i]], e)

            img = img.astype(np.float32)
            img /= 255.
            img -= 0.5
            img *= 2.
            img = img[:, :, ::-1]

            X[i, ::] = img
            y[i, :] = self.y_train[batch_ids[i]]
        
        return np.array(X), y
