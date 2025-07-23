# CS412 - Machine Learning Assignments

This repository contains the homework assignments completed for the **CS412 - Machine Learning** course during Spring 2024-25. The assignments provide hands-on experience with fundamental and advanced machine learning techniques, including supervised learning, optimization algorithms, and deep learning with transfer learning.

---

## 📚 Homework Overview

| Homework | Topic | Main Techniques |
|----------|-------|------------------|
| [HW1](#hw1---knn--decision-tree-classification-on-mnist) | MNIST Digit Classification | k-Nearest Neighbors, Decision Tree |
| [HW2](#hw2---linear--polynomial-regression) | Regression Analysis | OLS, Gradient Descent, Polynomial Regression |
| [HW3](#hw3---gradient-descent--naïve-bayes) | Optimization & Probabilistic Models | Steepest Descent, Naïve Bayes |
| [HW4](#hw4---transfer-learning-on-celeba) | Deep Learning with Transfer Learning | VGG-16, Fine-Tuning, Binary Classification |

---

## 🧠 HW1 – kNN & Decision Tree Classification on MNIST

**Objective:** Implement and compare k-NN and Decision Tree classifiers on the MNIST handwritten digit dataset.

- Data preprocessing and normalization
- Class distribution and statistical analysis
- Hyperparameter tuning (`k`, `max_depth`, `min_samples_split`)
- Evaluation using accuracy, precision, recall, F1-score
- Visualization: confusion matrices, ROC curves
- Misclassification analysis

📁 Files:  
- `CS412-HW1-HüseryinErenYildiz.ipynb`  
- `CS412-HW1-HüseryinErenYildiz.pdf`

---

## 📈 HW2 – Linear & Polynomial Regression

**Objective:** Implement regression methods on synthetic datasets to model linear and nonlinear relationships.

- Dataset 1: Linear function with Gaussian noise  
  - Scikit-learn Linear Regression  
  - Manual OLS (pseudoinverse)  
  - Manual Gradient Descent  
- Dataset 2: Nonlinear function  
  - Polynomial regression (degrees 1, 3, 5, 7)  
  - Manual implementation for degree 3
- Model evaluation via Mean Squared Error (MSE)
- Line fitting and loss curve visualization

📁 Files:  
- `CS412-HW2-HüseryinErenYildiz.ipynb`  
- `CS412-HW2-HüseryinErenYildiz.pdf`

---

## 🔁 HW3 – Gradient Descent & Naïve Bayes

**Objective:** Apply optimization techniques and implement a Naïve Bayes classifier from scratch.

- **Part 1:** Steepest descent to minimize `F(x, y) = x² + 4x + y² - 4y`
  - Manual gradient calculation
  - Iterative updates and function value tracking
- **Part 2:** Naïve Bayes classification on the PlayTennis dataset
  - With and without Laplace (Add-1) smoothing
  - Normalized posterior probability computation

📁 Files:  
- `CS412-HW3-HüseryinErenYildiz.pdf`

---

## 🤖 HW4 – Transfer Learning on CelebA

**Objective:** Perform binary gender classification using a pre-trained **VGG-16** model on the **CelebA** dataset.

- Dataset: 30,000 celebrity face images with gender labels
- Data split: 80% train, 10% validation, 10% test
- Two training strategies:
  1. Freeze all convolutional layers, train only classifier head
  2. Fine-tune last conv block + classifier head
- Two learning rates: 0.001 and 0.0001
- Evaluation metrics: Accuracy, Confusion Matrix
- Visualization of training loss, accuracy, and misclassified samples

📁 Files:  
- `CS412-HW4-HüseryinErenYildiz.ipynb`  
- `CS412-HW4-HüseryinErenYildiz.pdf`

---

## 📦 Dependencies

All notebooks are written in **Python 3.x** using **Google Colab** with the following major libraries:

- `NumPy`, `Pandas`
- `scikit-learn`
- `matplotlib`, `seaborn`
- `TensorFlow` / `Keras` (for HW4)

---

## 🧾 License

These assignments are submitted as part of the CS412 course at our university. Reuse or distribution without permission is not allowed.

---

