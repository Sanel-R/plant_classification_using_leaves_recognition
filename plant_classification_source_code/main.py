import cv2
from imutils import paths
from skimage.exposure import rescale_intensity
from skimage.io import imshow, imread, show
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu, threshold_yen
from skimage.morphology import closing
from skimage.measure import label, regionprops, regionprops_table
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm
import os

# we read the images that will be used for classification and learning.
read_train_images = list(list(paths.list_images(r'Classfier/Train')))

# define the dataframe.
df = pd.DataFrame()

for i in read_train_images:
    image_paths = i
    image = rgb2gray(imread(image_paths))
    binary = image < threshold_otsu(image)
    binary = closing(binary)
    label_img = label(binary)

    table = pd.DataFrame(regionprops_table(label_img, image, ['convex_area', 'area', 'eccentricity',
                                                              'extent', 'inertia_tensor',
                                                              'major_axis_length', 'minor_axis_length',
                                                              'perimeter', 'solidity', 'image',
                                                              'orientation', 'moments_central',
                                                              'moments_hu', 'euler_number',
                                                              'equivalent_diameter',
                                                              'mean_intensity', 'bbox']))

    table['perimeter_area_ratio'] = table['perimeter'] / table['area']

    real_images = []
    std = []
    mean = []
    percent25 = []
    percent75 = []

    for prop in regionprops(label_img):
        min_row, min_col, max_row, max_col = prop.bbox
        img = image[min_row:max_row, min_col:max_col]
        real_images += [img]
        mean += [np.mean(img)]
        std += [np.std(img)]
        percent25 += [np.percentile(img, 25)]
        percent75 += [np.percentile(img, 75)]

    table['real_images'] = real_images
    table['mean_intensity'] = mean
    table['std_intensity'] = std
    table['25th Percentile'] = mean
    table['75th Percentile'] = std
    table['iqr'] = table['75th Percentile'] - table['25th Percentile']

    table['label'] = image_paths.split('\\')[1]
    df = pd.concat([df, table], axis=0)

X = df.drop(columns=['label', 'image', 'real_images'])

# features
X = X[['iqr', '75th Percentile', 'inertia_tensor-1-1',
       'std_intensity', 'mean_intensity', '25th Percentile',
       'minor_axis_length', 'solidity', 'eccentricity']]

# target
y = df['label']
columns = X.columns

# train-test-split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=123, stratify=y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=123)

clf = GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=87)

clf.fit(X_train, y_train)

# print confusion matrix of test set
print(classification_report(clf.predict(X_test), y_test))

# print accuracy score of the test set
print(f"Test Accuracy: {np.mean(clf.predict(X_test) == y_test) * 100:.2f}%")
