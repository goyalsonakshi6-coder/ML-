#3.1

# Import libraries
from sklearn.neighbors import KNeighborsClassifier

# Training data (Weight, Size)
X = [
    [150, 7],
    [170, 7.5],
    [140, 6.8],
    [130, 6],
    [180, 8],
    [160, 7.2]
]

# Labels
# 0 = Apple
# 1 = Orange
y = [0, 1, 0, 0, 1, 1]

# Create KNN classifier
knn = KNeighborsClassifier(n_neighbors=3)

# Train the model
knn.fit(X, y)

# New fruit to classify
new_fruit = [[155, 7]]

# Predict
prediction = knn.predict(new_fruit)

# Display result
if prediction[0] == 0:
    print("Predicted Fruit: Apple")
else:
    print("Predicted Fruit: Orange")


#3.2

import numpy as np

# Training data
class_A = np.array([[2,3], [3,4], [2,5]])
class_B = np.array([[8,8], [9,9], [7,8]])

# Calculate centroids
centroid_A = np.mean(class_A, axis=0)
centroid_B = np.mean(class_B, axis=0)

# New point
new_point = np.array([3,5])

# Calculate distances
dist_A = np.linalg.norm(new_point - centroid_A)
dist_B = np.linalg.norm(new_point - centroid_B)

# Classify
if dist_A < dist_B:
    print("Class A")
else:
    print("Class B")


#  **KNN**                         | "Ask the **nearest K friends** what class they belong to."                           |
#  **Minimum Distance Classifier** | "Ask only the **class leader (centroid)** of each group and 
# choose the closest one." 

#3.3

# Type: Supervised Learning
# Purpose: Predict continuous values (salary, marks, house price, temperature)
# Equation: Y^ = B0+B1x
# Goal: Find the best-fit straight line.
# Training: Learns the slope and intercept by minimizing the squared prediction errors.
# Output: A numerical value (not a class label).

from sklearn.linear_model import LinearRegression
import numpy as np

# Training data
X = np.array([[1], [2], [3], [4], [5]])   # Study hours
y = np.array([30, 40, 50, 60, 70])        # Marks

# Create model
model = LinearRegression()

# Train model
model.fit(X, y)

# Predict marks for 6 study hours
prediction = model.predict([[6]])

print("Predicted Marks:", prediction[0])

#3.5

#Logistic Regression solves this problem using the Sigmoid Function.
#The sigmoid function converts any number into a value between 0 and 1.



#3.7

from sklearn.ensemble import RandomForestClassifier

# Training data
X = [
    [2, 60],
    [5, 90],
    [3, 70],
    [6, 95],
    [4, 80]
]

# Labels
y = ["Fail", "Pass", "Fail", "Pass", "Pass"]

# Create Random Forest model
model = RandomForestClassifier(n_estimators=100)

# Train
model.fit(X, y)

# Predict
prediction = model.predict([[5, 85]])

print(prediction)


#3.8

from sklearn.feature_extraction.text import CountVectorizer

documents = [
    "I love machine learning",
    "I love Python",
    "Machine learning is fun"
]

vectorizer = CountVectorizer()

X = vectorizer.fit_transform(documents)

print(vectorizer.get_feature_names_out())
print(X.toarray())


#3.9

# Feature Extraction = Convert raw data into useful information.
# Classification = Use that information to predict a category.

#Without feature extraction
# Image
# 🐱
# Computer sees
# Millions of pixels
# Very difficult.
# Feature extraction converts it into meaningful numbers.

# Example
# Instead of all pixels

# Number of edges
# Average brightness
# Texture
# Histogram
# Corners

# These numbers are much more useful.
# Classifier becomes much more accurate.

#2.10

from sklearn.decomposition import PCA
from sklearn.datasets import load_digits

# Load dataset
digits = load_digits()

X = digits.data

# Create PCA
pca = PCA(n_components=20)

# Reduce features
X_new = pca.fit_transform(X)

print(X.shape)
print(X_new.shape)

# PCA is not a classifier.
# It only reduces features.

#M2
import cv2

img = cv2.imread("cat.jpg", 0)      # Read image
img = cv2.resize(img, (64, 64))     # Resize
features = img.flatten()            # Convert image to a feature vector

X = [features]                      # Store features

#3..6 ==> SVM
# Kernels allow SVM to handle non-linear data.

# 3.11
# Why is it called K-Means?
# K → Number of clusters.
# Means → The centroid is the mean (average) of the points in each cluster.

from sklearn.cluster import KMeans
import numpy as np

# Data
X = np.array([
    [1,2],
    [2,3],
    [3,4],
    [10,10],
    [11,11],
    [12,12]
])

# Create model
model = KMeans(n_clusters=2, random_state=42)

# Train
model.fit(X)

# Cluster number of each point
print(model.labels_)

# Centroids
print(model.cluster_centers_)

#3.12
# The time complexity of the K-Medoids algorithm is: 
# O(k×(nk)2)O(k×(n−k) 2)

#3.13

from sklearn.cluster import AgglomerativeClustering
import numpy as np

# Data
X = np.array([
    [1,2],
    [2,3],
    [3,4],
    [10,10],
    [11,11],
    [12,12]
])

# Create model
model = AgglomerativeClustering(n_clusters=2)

# Train
labels = model.fit_predict(X)

print(labels)

#3.14
# DBSCAN
from sklearn.cluster import DBSCAN
import numpy as np

# Data
X = np.array([
    [1,2],
    [2,2],
    [2,3],
    [8,8],
    [8,9],
    [25,25]
])

# Create model
model = DBSCAN(eps=2, min_samples=2)

# Train
labels = model.fit_predict(X)

print(labels)