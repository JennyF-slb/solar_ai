# Imports
import os
import numpy as np
import pandas as pd
import matplotlib as plt
import rasterio
from osgeo import gdal
import tensorflow
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image

# Defining class
class Preprocessing:
    # Function to view folder tree
    def list_folder_tree(folder_path):
        for root, dirs, _ in os.walk(folder_path):
            if os.path.basename(root) != '.ipynb_checkpoints':
                level = root.replace(folder_path, '').count(os.sep)
                indent = ' ' * 4 * (level)
                print('{}{}/'.format(indent, os.path.basename(root)))


    # Function to load and preprocess GeoTIFF files
    def load_geotiff_data(image_folder, target_shape=None, normalize=True):
        file_paths = [os.path.join(image_folder, filename) for filename in sorted(os.listdir(image_folder)) if filename.endswith('.tif')]
        data = []
        for file_path in file_paths:
            dataset = rasterio.open(file_path, "r")
            if dataset is None:
                continue
            image = dataset.read() # Convert to array
            image = np.transpose(image, (1, 2, 0))  # Transpose the position of channels in the image

            if target_shape is not None:
                image = tensorflow.image.resize(image, target_shape[:2]) # Re-shape / re-size the image
                image = image.numpy()
            else:
                target_shape = (image.shape[0], image.shape[1])  # Use the original image shape as the target shape
                image = tensorflow.image.resize(image, target_shape[:2])
                image = image.numpy()

            image = image.astype('float32') / 255.0 # Normalize pixel color values between 0 and 1 from 0 and 255

            # Add the preprocessed image to the data list
            data.append(image)
            dataset.close()  # Close the dataset

        return np.array(data)

    # Funtion to plot loss and accuracy
    def plot_loss_accuracy(history, title=None):
        fig, ax = plt.subplots(1,2, figsize=(20,7))

        # --- LOSS ---

        ax[0].plot(history.history['loss'])
        ax[0].plot(history.history['val_loss'])
        ax[0].set_title('Model loss')
        ax[0].set_ylabel('Loss')
        ax[0].set_xlabel('Epoch')
        ax[0].legend(['Train', 'Val'], loc='best')
        ax[0].grid(axis="x",linewidth=0.5)
        ax[0].grid(axis="y",linewidth=0.5)

        # --- ACCURACY

        ax[1].plot(history.history['accuracy'])
        ax[1].plot(history.history['val_accuracy'])
        ax[1].set_title('Model Accuracy')
        ax[1].set_ylabel('Accuracy')
        ax[1].set_xlabel('Epoch')
        ax[1].legend(['Train', 'Val'], loc='best')
        ax[1].grid(axis="x",linewidth=0.5)
        ax[1].grid(axis="y",linewidth=0.5)

        if title:
            fig.suptitle(title)


    # Function to extract metadata from GeoTIFF files
    def extract_metadata(folder_path):
        # Create an empty list to store the metadata
        metadata_list = []

        # Iterate over each file in the folder
        for filename in os.listdir(folder_path):
            if filename.endswith('.tif') or filename.endswith('.tiff'):
                # Create the file path for the current GeoTIFF file
                file_path = os.path.join(folder_path, filename)

                try:
                    # Open the GeoTIFF file
                    dataset = gdal.Open(file_path)

                    # Get the metadata
                    width = dataset.RasterXSize
                    height = dataset.RasterYSize
                    crs = dataset.GetProjection()

                    # Get the geotransform
                    geotransform = dataset.GetGeoTransform()
                    origin_x = geotransform[0]
                    origin_y = geotransform[3]
                    pixel_width = geotransform[1]
                    pixel_height = geotransform[5]
                    rotation_x = geotransform[2]
                    rotation_y = geotransform[4]

                    # Append the metadata to the list
                    metadata_list.append({
                        'File': filename,
                        'Width': width,
                        'Height': height,
                        'CRS': crs,
                        'Origin_X': origin_x,
                        'Origin_Y': origin_y,
                        'Pixel_Width': pixel_width,
                        'Pixel_Height': pixel_height,
                        'Rotation_X': rotation_x,
                        'Rotation_Y': rotation_y
                    })

                    # Close the dataset
                    dataset = None

                except Exception as e:
                    print(f"Error opening file: {file_path}")
                    print(f"Error message: {str(e)}")

        # Create a dataframe from the metadata list
        metadata_df = pd.DataFrame(metadata_list, columns=['File', 'Width', 'Height', 'CRS', 'Origin_X', 'Origin_Y',
                                                        'Pixel_Width', 'Pixel_Height', 'Rotation_X', 'Rotation_Y'])

        return metadata_df

    # Convert Geo-TIF files to PNG and save in the output folder
    def convert_tif_to_png(tif_folder_path, png_output_folder_path):
        for filename in os.listdir(tif_folder_path):
            if filename.endswith(".tif"):
                tif_path = os.path.join(tif_folder_path, filename)
                png_filename = filename[:-4] + ".png"
                png_path = os.path.join(png_output_folder_path, png_filename)
                img = Image.open(tif_path)
                img.save(png_path, "PNG")

    # Function to create patches from image
    def create_patches(image, patch_size, overlap): #image is image array with dimention, e.g, (5000,5000,3). patch_size is 2d. overlap is a number
        height, width = image.shape[:2]
        patch_height, patch_width = patch_size

        stride_height = patch_height - overlap
        stride_width = patch_width - overlap

        patches = []

        for y in range(0, height-patch_height+1, stride_height):
            for x in range(0, width-patch_width+1, stride_width):
                patch = image[y:y+patch_height, x:x+patch_width]
                patches.append(patch)

        return patches   #return all patches

    # Function to generate data / augmentations
    def get_train_datagen():
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            samplewise_std_normalization=False,
            horizontal_flip=True,
            vertical_flip=False,
            height_shift_range=0.1,
            width_shift_range=0.1,
            rotation_range=3,
            shear_range=0.01,
            fill_mode='nearest',
            zoom_range=0.05,
            zca_whitening=True,
            zca_epsilon=1e-5
            # preprocessing_function=patch_image
        )
        return train_datagen
