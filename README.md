# Kannada_MNIST_Classification_Problem

Problem Statement
This project aims to solve a classification problem using the Kannada MNIST dataset, which consists of handwritten digits in the Kannada script. The goal is to classify these digits into one of the 10 classes. The dataset can be downloaded from the following link: [Kannada MNIST Dataset.](https://www.kaggle.com/datasets/higgstachyon/kannada-mnist)

Dataset
The Kannada MNIST dataset contains 60,000 images for training and 10,000 images for testing. Each image is of size 28x28 pixels. To preprocess the dataset, the following steps will be performed:

Extract the dataset from the downloaded .npz file or from the web.
Perform Principal Component Analysis (PCA) to reduce the dimensionality of the images to 10 components. This will transform the images from the original 28x28 dimensions to 10 dimensions.
Models and Evaluation

The following models will be applied to the preprocessed dataset:

Decision Trees
Random Forest
Naive Bayes Model
K-NN Classifier
Support Vector Machines (SVM)

For each model, the following metrics will be computed:

Precision
Recall
F1-Score
Confusion Matrix
ROC-AUC Curve

Example: 
RESULTS FOR DECISION TREE WITH 30 COMPONENTS:
Classification Report:
              precision    recall  f1-score   support

           0       0.78      0.73      0.76      1000
           1       0.76      0.79      0.77      1000
           2       0.93      0.93      0.93      1000
           3       0.77      0.80      0.78      1000
           4       0.80      0.84      0.82      1000
           5       0.81      0.80      0.80      1000
           6       0.78      0.80      0.79      1000
           7       0.77      0.65      0.70      1000
           8       0.83      0.89      0.86      1000
           9       0.81      0.83      0.82      1000

    accuracy                           0.80     10000
   macro avg       0.80      0.80      0.80     10000
weighted avg       0.80      0.80      0.80     10000



Confusion Matrix:
[[729 153  12  23   9   4   9   9  35  17]
 [106 786  10  34   7  10   7   9  19  12]
 [  3   1 931   8   0  25   7  15   4   6]
 [ 13  12  17 795  19  24  31  68  10  11]
 [  1   4   1  40 841  52  12  12  16  21]
 [ 17   9   8  26  95 800   6   7  17  15]
 [  6   6   7  41  18  32 798  69   5  18]
 [ 13  11   7  53  28   8 137 648   8  87]
 [ 29  25   9   2  10  20   1   2 890  12]
 [ 13  33   4   5  22  13  10   5  65 830]]

 ROC_AUC:![image](https://github.com/NAVEEN-RAJ-C/Kannada_MNIST_Classification_Problem/assets/133734968/72f8f6ad-5fd3-4234-b547-032358b8f383)

Conclusion
By following the steps outlined in the Jupyter Notebook, you will be able to preprocess the Kannada MNIST dataset, apply various classification models, and evaluate their performance using different metrics. The results obtained will help in understanding the effectiveness of each model for the given classification problem.
