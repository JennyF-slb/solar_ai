# Imports
import os
import numpy as np
import pandas as pd

import tensorflow

# Defining class
class Metrics:
    def jaccard_similarity_loss(y_true, y_pred):
        intersection = tensorflow.reduce_sum(y_true * y_pred, axis=[1, 2])
        union = tensorflow.reduce_sum(y_true + y_pred, axis=[1, 2]) - intersection
        jaccard = intersection / (union + tensorflow.keras.backend.epsilon())
        loss = 1 - jaccard
        return loss
