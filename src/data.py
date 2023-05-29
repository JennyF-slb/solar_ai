import os
import pandas as pd

class Inria:
    def get_data(self):
        """
        This function returns a Python dict containing file names from specific directories.
        """
        BASE_PATH = os.path.dirname(os.path.dirname(__file__))
        data_path = os.path.join(BASE_PATH, "data/AerialImageDataset")

        # File paths
        train_images_dir = os.path.join(data_path, 'train', 'images')
        test_images_dir = os.path.join(data_path, 'test', 'images')
        train_gt_dir = os.path.join(data_path, 'train', 'gt')

        file_names = {
            'train_images': [f for f in os.listdir(train_images_dir) if f.endswith('.tif')],
            'test_images': [f for f in os.listdir(test_images_dir) if f.endswith('.tif')],
            'train_gt': [f for f in os.listdir(train_gt_dir) if f.endswith('.tif')]
        }
        return file_names
