# Imports
import os
import numpy as np
import pandas as pd

import tensorflow
from tensorflow.keras.preprocessing.image import array_to_img, img_to_array, load_img

# Defining class
class Prediction:
    def read_file(directory):
        blind_test = []
        for dirname, _, filenames in os.walk(directory):
            for filename in filenames:
                blind_test.append(os.path.join(dirname, filename))
        blind_test = sorted(blind_test)
        return blind_test

    # Build the dataset using list of filenames and target image size
    def build_dataset(dir_list, img_size):
        num_imgs = len(dir_list)
        test_imgs = np.zeros((num_imgs,) + img_size + (3,), dtype="float32")

        for i in range(len(dir_list)):
            test_imgs[i] = img_to_array(load_img(dir_list[i], target_size=img_size))
        return test_imgs
