import os
import numpy as np
np.warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import pandas as pd
from tqdm import tqdm

from google_drive_downloader import GoogleDriveDownloader as gdd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.contrib.eager.python import tfe
from PIL import Image

tf.enable_eager_execution()
tf.set_random_seed(0)
np.random.seed(0)

# gdd.download_file_from_google_drive(file_id='1ycR7Aexe8xbZ8oEDsQwGc9SIiFklRpfu', dest_path='./data.zip', unzip=True)

data_dir = 'data'
os.listdir(data_dir)

#Read data
train_df = pd.read_csv(os.path.join(data_dir, 'train.csv'))
train_df.head()

train_df.info()

test_df = pd.read_csv(os.path.join(data_dir, 'sample_submission.csv'))
test_df.head()

test_df.info()

train_df.label.value_counts()

"""
TODO 1: Cài đặt hàm đọc ảnh và đưa về NumPy Array
"""

#generate_data to create train and test matrix
def generate_data(image_paths, size=224):
    """
    Đọc và chuyển các ảnh về numpy array
    
    Parameters
    ----------
    image_paths: list of N strings
        List các đường dẫn ảnh
    size: int
        Kích thước ảnh cần resize
    
    Returns
    -------
    numpy array kích thước (N, size, size, 3)
    """
    image_array = np.zeros((len(image_paths), size, size, 3), dtype='float32')
    
    for idx, image_path in tqdm(enumerate(image_paths)):
        ### START CODE HERE
        
        # Đọc ảnh bằng thư viện Pillow và resize ảnh
        image = Image.open(image_path)
        image = image.resize()
        
        # Chuyển ảnh thành numpy array và gán lại mảng image_array
        image_array
        
        ### END CODE HERE
    return image_array

# List các đường dẫn file cho việc huấn luyện
train_files = [os.path.join("data/images", file) for file in train_df.image]

# List các nhãn
train_y = train_df.label

# Tạo numpy array cho dữ liệu huấn luyện
train_arr = generate_data(train_files)

train_arr.shape

#Tao tensor cho du lieu test

test_files = [os.path.join("data/images", file) for file in test_df.image]
test_x = generate_data(test_files)
test_x.shape

#Tao one-hot cho train_y

num_classes = len(np.unique(train_y))
y_ohe = tf.keras.utils.to_categorical(train_y, num_classes=num_classes)

#Chia du lieu de huan luyen va danh gia
x_train, x_valid, y_train_ohe, y_valid_ohe = train_test_split(train_arr, y_ohe, test_size=0.25)

print("Train size: {} - Validation size: {}".format(x_train.shape, x_valid.shape))