# https://youtu.be/5x-CIHRmMNY
"""
@author: Sreenivas Bhattiprolu
skimage.feature.greycomatrix(image, distances, angles, levels=None, symmetric=False, normed=False)
distances - List of pixel pair distance offsets.
angles - List of pixel pair angles in radians.
skimage.feature.greycoprops(P, prop)
prop: The property of the GLCM to compute.
{‘contrast’, ‘dissimilarity’, ‘homogeneity’, ‘energy’, ‘correlation’, ‘ASM’}
"""

import numpy as np
import matplotlib.pyplot as plt
import glob
import cv2
import os
import pandas as pd
from skimage.filters import sobel
import seaborn as sns
from skimage.feature import greycomatrix, greycoprops
from skimage.measure import shannon_entropy
from imutils import paths

# read files for processing.
readTrainImages = list(paths.list_images(
    r"C:\Users\Sanele\Documents\2021 Computer Science Hons Directory\Semester 1\COMP702 - Image processing and Computer vision\Project due 04 June 2021\Classfier\Train"))
readValidationImages = list(paths.list_images(
    r"C:\Users\Sanele\Documents\2021 Computer Science Hons Directory\Semester 1\COMP702 - Image processing and Computer vision\Project due 04 June 2021\Classfier\Compare"))

# readTrainImages = list(paths.list_images(
#     r"C:\Users\Sanele\Documents\2021 Computer Science Hons Directory\Semester 1\COMP702 - Image processing and Computer vision\Project due 04 June 2021\Dataset - leaf recognition\dataset\images\field"))
# readValidationImages = list(paths.list_images(
#     r"C:\Users\Sanele\Documents\2021 Computer Science Hons Directory\Semester 1\COMP702 - Image processing and Computer vision\Project due 04 June 2021\Dataset - leaf recognition\dataset\images\lab"))

# Resize images to
SIZE = 750
W_SIZE = 690
H_SIZE = 735

# Capture images and labels into arrays.
# Start by creating empty lists.
train_images = []
train_labels = []
# for directory_path in glob.glob("cell_images/train/*"):
for img_paths in readTrainImages:
    label = img_paths.split("\\")
    img = cv2.imread(img_paths)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (W_SIZE, H_SIZE))
    train_labels.append(label[len(label) - 2])
    train_images.append(img)

train_images = np.array(train_images)
train_labels = np.array(train_labels)

# Do exactly the same for test/validation images
# test
test_images = []
test_labels = []
# for directory_path in glob.glob("cell_images/test/*"):
for img_paths in readValidationImages:
    label = img_paths.split("\\")
    img = cv2.imread(img_paths)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (W_SIZE, H_SIZE))
    test_labels.append(label[len(label) - 2])
    test_images.append(img)

test_images = np.array(test_images)
test_labels = np.array(test_labels)

# Encode labels from text (folder names) to integers.
from sklearn import preprocessing

le = preprocessing.LabelEncoder()
le.fit(test_labels)
test_labels_encoded = le.transform(test_labels)
le.fit(train_labels)
train_labels_encoded = le.transform(train_labels)

# Split data into test and train datasets (already split but assigning to meaningful convention)
# If you only have one dataset then split here
x_train, y_train, x_test, y_test = train_images, train_labels_encoded, test_images, test_labels_encoded


# Normalize pixel values to between 0 and 1
# x_train, x_test = x_train / 255.0, x_test / 255.0

###################################################################
# FEATURE EXTRACTOR function
# input shape is (n, x, y, c) - number of images, x, y, and channels
# for this specific function, we change the distance when reading our data cause the accuracy is power.
def feature_extractor(dataset):
    image_dataset = pd.DataFrame()
    for image in range(dataset.shape[0]):  # iterate through each file
        # print(image)

        df = pd.DataFrame()  # Temporary data frame to capture information for each loop.
        # Reset dataframe to blank after each loop.

        img = dataset[image, :, :]
        ################################################################
        # START ADDING DATA TO THE DATAFRAME

        # Full image
        # GLCM = greycomatrix(img, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4])
        GLCM = greycomatrix(img, [1], [0])
        GLCM_Energy = greycoprops(GLCM, 'energy')[0]
        df['Energy'] = GLCM_Energy
        GLCM_corr = greycoprops(GLCM, 'correlation')[0]
        df['Corr'] = GLCM_corr
        GLCM_diss = greycoprops(GLCM, 'dissimilarity')[0]
        df['Diss_sim'] = GLCM_diss
        GLCM_hom = greycoprops(GLCM, 'homogeneity')[0]
        df['Homogen'] = GLCM_hom
        GLCM_contr = greycoprops(GLCM, 'contrast')[0]
        df['Contrast'] = GLCM_contr
        GLCM_asm = greycoprops(GLCM, 'ASM')
        df['ASM'] = GLCM_asm

        GLCM2 = greycomatrix(img, [3], [0])
        GLCM_Energy2 = greycoprops(GLCM2, 'energy')[0]
        df['Energy2'] = GLCM_Energy2
        GLCM_corr2 = greycoprops(GLCM2, 'correlation')[0]
        df['Corr2'] = GLCM_corr2
        GLCM_diss2 = greycoprops(GLCM2, 'dissimilarity')[0]
        df['Diss_sim2'] = GLCM_diss2
        GLCM_hom2 = greycoprops(GLCM2, 'homogeneity')[0]
        df['Homogen2'] = GLCM_hom2
        GLCM_contr2 = greycoprops(GLCM2, 'contrast')[0]
        df['Contrast2'] = GLCM_contr2
        GLCM_asm2 = greycoprops(GLCM2, 'ASM')
        df['ASM'] = GLCM_asm2

        GLCM3 = greycomatrix(img, [5], [0])
        GLCM_Energy3 = greycoprops(GLCM3, 'energy')[0]
        df['Energy3'] = GLCM_Energy3
        GLCM_corr3 = greycoprops(GLCM3, 'correlation')[0]
        df['Corr3'] = GLCM_corr3
        GLCM_diss3 = greycoprops(GLCM3, 'dissimilarity')[0]
        df['Diss_sim3'] = GLCM_diss3
        GLCM_hom3 = greycoprops(GLCM3, 'homogeneity')[0]
        df['Homogen3'] = GLCM_hom3
        GLCM_contr3 = greycoprops(GLCM3, 'contrast')[0]
        df['Contrast3'] = GLCM_contr3
        GLCM_asm3 = greycoprops(GLCM3, 'ASM')
        df['ASM'] = GLCM_asm3

        GLCM4 = greycomatrix(img, [0], [np.pi / 4])
        GLCM_Energy4 = greycoprops(GLCM4, 'energy')[0]
        df['Energy4'] = GLCM_Energy4
        GLCM_corr4 = greycoprops(GLCM4, 'correlation')[0]
        df['Corr4'] = GLCM_corr4
        GLCM_diss4 = greycoprops(GLCM4, 'dissimilarity')[0]
        df['Diss_sim4'] = GLCM_diss4
        GLCM_hom4 = greycoprops(GLCM4, 'homogeneity')[0]
        df['Homogen4'] = GLCM_hom4
        GLCM_contr4 = greycoprops(GLCM4, 'contrast')[0]
        df['Contrast4'] = GLCM_contr4
        GLCM_asm4 = greycoprops(GLCM4, 'ASM')
        df['ASM'] = GLCM_asm4

        # we add the following snippets to assist in the pattern recognition process.
        GLCM5 = greycomatrix(img, [1], [np.pi / 4])
        GLCM_Energy5 = greycoprops(GLCM5, 'energy')[0]
        df['Energy5'] = GLCM_Energy5
        GLCM_corr5 = greycoprops(GLCM5, 'correlation')[0]
        df['Corr5'] = GLCM_corr5
        GLCM_diss5 = greycoprops(GLCM5, 'dissimilarity')[0]
        df['Diss_sim5'] = GLCM_diss5
        GLCM_hom5 = greycoprops(GLCM5, 'homogeneity')[0]
        df['Homogen5'] = GLCM_hom5
        GLCM_contr5 = greycoprops(GLCM5, 'contrast')[0]
        df['Contrast5'] = GLCM_contr5
        GLCM_asm5 = greycoprops(GLCM5, 'ASM')
        df['ASM'] = GLCM_asm5

        GLCM6 = greycomatrix(img, [3], [np.pi / 4])
        GLCM_Energy6 = greycoprops(GLCM6, 'energy')[0]
        df['Energy6'] = GLCM_Energy6
        GLCM_corr6 = greycoprops(GLCM6, 'correlation')[0]
        df['Corr6'] = GLCM_corr6
        GLCM_diss6 = greycoprops(GLCM6, 'dissimilarity')[0]
        df['Diss_sim6'] = GLCM_diss6
        GLCM_hom6 = greycoprops(GLCM6, 'homogeneity')[0]
        df['Homogen6'] = GLCM_hom6
        GLCM_contr6 = greycoprops(GLCM6, 'contrast')[0]
        df['Contrast6'] = GLCM_contr6
        GLCM_asm6 = greycoprops(GLCM6, 'ASM')
        df['ASM'] = GLCM_asm6

        GLCM7 = greycomatrix(img, [5], [np.pi / 4])
        GLCM_Energy7 = greycoprops(GLCM7, 'energy')[0]
        df['Energy7'] = GLCM_Energy7
        GLCM_corr7 = greycoprops(GLCM7, 'correlation')[0]
        df['Corr7'] = GLCM_corr7
        GLCM_diss7 = greycoprops(GLCM7, 'dissimilarity')[0]
        df['Diss_sim7'] = GLCM_diss7
        GLCM_hom7 = greycoprops(GLCM7, 'homogeneity')[0]
        df['Homogen7'] = GLCM_hom7
        GLCM_contr7 = greycoprops(GLCM7, 'contrast')[0]
        df['Contrast7'] = GLCM_contr7
        GLCM_asm7 = greycoprops(GLCM7, 'ASM')
        df['ASM'] = GLCM_asm7

        GLCM8 = greycomatrix(img, [1], [np.pi / 2])
        GLCM_Energy8 = greycoprops(GLCM8, 'energy')[0]
        df['Energy8'] = GLCM_Energy8
        GLCM_corr8 = greycoprops(GLCM8, 'correlation')[0]
        df['Corr8'] = GLCM_corr8
        GLCM_diss8 = greycoprops(GLCM8, 'dissimilarity')[0]
        df['Diss_sim8'] = GLCM_diss8
        GLCM_hom8 = greycoprops(GLCM8, 'homogeneity')[0]
        df['Homogen8'] = GLCM_hom8
        GLCM_contr8 = greycoprops(GLCM8, 'contrast')[0]
        df['Contrast8'] = GLCM_contr8
        GLCM_asm8 = greycoprops(GLCM8, 'ASM')
        df['ASM'] = GLCM_asm8

        GLCM9 = greycomatrix(img, [3], [np.pi / 2])
        GLCM_Energy9 = greycoprops(GLCM9, 'energy')[0]
        df['Energy9'] = GLCM_Energy9
        GLCM_corr9 = greycoprops(GLCM9, 'correlation')[0]
        df['Corr9'] = GLCM_corr9
        GLCM_diss9 = greycoprops(GLCM9, 'dissimilarity')[0]
        df['Diss_sim9'] = GLCM_diss9
        GLCM_hom9 = greycoprops(GLCM9, 'homogeneity')[0]
        df['Homogen9'] = GLCM_hom9
        GLCM_contr9 = greycoprops(GLCM9, 'contrast')[0]
        df['Contrast9'] = GLCM_contr9
        GLCM_asm9 = greycoprops(GLCM9, 'ASM')
        df['ASM'] = GLCM_asm9

        GLCM10 = greycomatrix(img, [5], [np.pi / 2])
        GLCM_Energy10 = greycoprops(GLCM10, 'energy')[0]
        df['Energy10'] = GLCM_Energy10
        GLCM_corr10 = greycoprops(GLCM10, 'correlation')[0]
        df['Corr10'] = GLCM_corr10
        GLCM_diss10 = greycoprops(GLCM10, 'dissimilarity')[0]
        df['Diss_sim10'] = GLCM_diss10
        GLCM_hom10 = greycoprops(GLCM10, 'homogeneity')[0]
        df['Homogen10'] = GLCM_hom10
        GLCM_contr10 = greycoprops(GLCM10, 'contrast')[0]
        df['Contrast10'] = GLCM_contr10
        GLCM_asm10 = greycoprops(GLCM10, 'ASM')
        df['ASM'] = GLCM_asm10

        # Add more filters as needed
        # entropy = shannon_entropy(img)
        # df['Entropy'] = entropy

        # Append features from current image to the dataset
        image_dataset = image_dataset.append(df)

    return image_dataset


####################################################################
# Extract features from training images
image_features = feature_extractor(x_train)
X_for_ML = image_features
# Reshape to a vector for Random Forest / SVM training
# n_features = image_features.shape[1]
# image_features = np.expand_dims(image_features, axis=0)
# X_for_ML = np.reshape(image_features, (x_train.shape[0], -1))  #Reshape to #images, features

# Define the classifier
# from sklearn.ensemble import RandomForestClassifier
# RF_model = RandomForestClassifier(n_estimators = 50, random_state = 42)

# Can also use SVM but RF is faster and may be more accurate.
# from sklearn import svm
# SVM_model = svm.SVC(decision_function_shape='ovo')  #For multiclass classification
# SVM_model.fit(X_for_ML, y_train)

# Fit the model on training data
# RF_model.fit(X_for_ML, y_train) #For sklearn no one hot encoding


import lightgbm as lgb

# Class names for LGBM start at 0 so reassigning labels from 1,2,3,4 to 0,1,2,3
d_train = lgb.Dataset(X_for_ML, label=y_train)

# https://lightgbm.readthedocs.io/en/latest/Parameters.html
lgbm_params = {'task': 'train',
               'boosting_type': 'gbdt',
               'objective': 'multiclass',
               'num_class': 5,
               'metric': 'multi_logloss',
               'learning_rate': 0.002296,
               'max_depth': 7,
               'num_leaves': 17,
               'feature_fraction': 0.4,
               'bagging_fraction': 0.6,
               'bagging_freq': 17}  # no.of unique values in the target class not inclusive of the end value

lgb_model = lgb.train(lgbm_params, d_train, 50)  # 50 iterations. Increase iterations for small learning rates

# Predict on Test data
# Extract features from test data and reshape, just like training data
test_features = feature_extractor(x_test)
test_features = np.expand_dims(test_features, axis=0)
test_for_RF = np.reshape(test_features, (x_test.shape[0], -1))

# Predict on test
test_prediction = lgb_model.predict(test_for_RF)
test_prediction = np.argmax(test_prediction, axis=1)
# Inverse le transform to get original label back.
test_prediction = le.inverse_transform(test_prediction)

# Print overall accuracy
from sklearn import metrics

print("Accuracy = ", metrics.accuracy_score(test_labels, test_prediction))

# Print confusion matrix
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(test_labels, test_prediction)

fig, ax = plt.subplots(figsize=(6, 6))  # Sample figsize in inches
sns.set(font_scale=1.6)
sns.heatmap(cm, annot=True, linewidths=.5, ax=ax)

# # Check results on a few random images
# import random
#
# n = random.randint(0, x_test.shape[0] - 1)  # Select the index of image to be loaded for testing
# img = x_test[n]
# plt.imshow(img)
#
# # Extract features and reshape to right dimensions
# input_img = np.expand_dims(img, axis=0)  # Expand dims so the input is (num images, x, y, c)
# input_img_features = feature_extractor(input_img)
# input_img_features = np.expand_dims(input_img_features, axis=0)
# input_img_for_RF = np.reshape(input_img_features, (input_img.shape[0], -1))
# # Predict
# img_prediction = lgb_model.predict(input_img_for_RF)
# img_prediction = np.argmax(img_prediction, axis=1)
# # img_prediction = le.inverse_transform([img_prediction])  # Reverse the label encoder to original name
# # print("The prediction for this image is: ", img_prediction)
# # print("The actual label for this image is: ", test_labels[n])
