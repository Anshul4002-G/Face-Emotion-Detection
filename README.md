# Face-Emotion-Detection
A machine learning model that will detect different emotion's of the user based on his mood and expression's/.
User is record to choose different face emotion's for both trainning and testing dataset . Which will help the model to get trainnned efficiently.
   
from keras.utils import to_categorical
from keras.preprocessing.image import load_img
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
import os
import pandas as pd
import numpy as np

train_dir='images/train'
test_dir='images/test'

def createdataframe(dir):
    image_paths=[]
    labels=[]
    for label in os.listdir(dir):
        for imagename in os.listdir(os.path.join(dir,label)):
            image_paths.append(os.path.join(dir,label,imagename))
            labels.append(label)
        print(label,"completed")
    return image_paths,labels


train=pd.DataFrame()
train['image'] ,train['label']=createdataframe(train_dir)
print(train)

test=pd.DataFrame()
test['images'],test['label']=createdataframe(test_dir)

from tqdm.notebook import tqdm

print(test)


def extract_features(images):
    features = []
    for image in tqdm(images):
        img = load_img(image, grayscale=True)
        img = np.array(img)
        features.append(img)  # Corrected: removed trailing comma
    features = np.array(features)  # Corrected: removed trailing comma
    features = features.reshape(len(features), 48, 48, 1)
    return features

train_features=extract_features(train['image'])

test_features=extract_features(test['images'])

x_train = train_features/255.0
x_test=test_features/255.0
