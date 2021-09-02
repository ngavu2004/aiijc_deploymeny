import pandas as pd
import numpy as np
import model
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
# from tensorflow.keras.applications import ResNet50
from PIL import Image
import json
import tensorflow.keras.backend as K
import cv2
import os

def preprocess_image(img_path,size=32):
    img= Image.open(img_path).convert('RGB')
    img = img.resize((size,size))
    img_array = img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch)
    img_array =img_array/255.0
    return img_array

def load_index_to_label_dict(path='index_to_class_label.json'):
    """Retrieves and formats the index to class label lookup dictionary needed to 
    make sense of the predictions. When loaded in, the keys are strings, this also
    processes those keys to integers."""
    with open(path, 'r') as f:
        index_to_class_label_dict = json.load(f)
    index_to_class_label_dict = {int(k): v for k, v in index_to_class_label_dict.items()}
    return index_to_class_label_dict

def predict(image,model):
    predictions = model.predict(image)    
    return predictions

def load_model(path='ResNet50.h5'):
    model = tf.keras.models.load_model(path, compile=False)
    # model.make_predict_function()
    return model

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

