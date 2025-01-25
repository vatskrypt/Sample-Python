import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Load the data
X = np.loadtxt('logisticX.csv', delimiter=',')
Y = np.loadtxt('logisticY.csv', delimiter=',')

# Step 2: Normalize the features (mean = 0, std = 1)
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
m, n = X.shape  # m = number of examples, n = number of features

# Add intercept term (bias) to X
X = np.hstack((np.ones((m, 1)), X))  # Shape becomes (m, n+1)

# Step 3: Initialize parameters
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_cost(X, Y, theta):
    m = len(Y)
    h = sigmoid(np.dot(X, theta))
    cost = -1/m * np.sum(Y * np.log(h) + (1 - Y) * np.log(1 - h))
    return cost

def gradient_descent(X, Y, theta, learning_rate, num_iters=10000, tol=1e-6):
    m = len(Y)
    cost_history = []
    for i in range(num_iters):
        h = sigmoid(np.dot(X, theta))
        gradient = np.dot(X.T, (h - Y)) / m
        theta -= learning_rate * gradient
        cost = compute_cost(X, Y, theta)
        cost_history.append(cost)
        if i > 0 and abs(cost_history[-1] - cost_history[-2]) < tol:
            print(f"Converged after {i+1} iterations.")
            break
    return theta, cost_history

# Step 4: Train and plot cost function vs iterations
learning_rate = 0.1
initial_theta = np.zeros(n + 1)
theta_1, cost_history_1 = gradient_descent(X, Y, initial_theta, learning_rate)

print(f"Final cost value: {cost_history_1[-1]}")
print(f"Learned parameters: {theta_1}")


# Plot cost vs iterations (for question 2)
plt.plot(range(len(cost_history_1)), cost_history_1, label=f"Learning Rate {learning_rate}")
plt.xlabel("Iterations")
plt.ylabel("Cost Function Value")
plt.title("Cost Function vs Iterations")
plt.legend()
plt.show()

# Step 5: Plot dataset and decision boundary (for question 3)
plt.plot(X[Y == 0][:, 1], X[Y == 0][:, 2], 'bo', label='Class 0')
plt.plot(X[Y == 1][:, 1], X[Y == 1][:, 2], 'ro', label='Class 1')

x_boundary = np.linspace(np.min(X[:, 1]), np.max(X[:, 1]), 100)
y_boundary = -(theta_1[0] + theta_1[1] * x_boundary) / theta_1[2]
plt.plot(x_boundary, y_boundary, color='green', label='Decision Boundary')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.title("Dataset and Decision Boundary")
plt.show()

# Step 6: Train with two learning rates (for question 4)
learning_rate_1 = 0.1
learning_rate_2 = 5

# Train for 100 iterations for both learning rates
theta_lr1, cost_history_lr1 = gradient_descent(X, Y, initial_theta, learning_rate_1, num_iters=100)
theta_lr2, cost_history_lr2 = gradient_descent(X, Y, initial_theta, learning_rate_2, num_iters=100)

# Plot cost vs iterations for both learning rates
plt.plot(range(len(cost_history_lr1)), cost_history_lr1, label=f"Learning Rate {learning_rate_1}")
plt.plot(range(len(cost_history_lr2)), cost_history_lr2, label=f"Learning Rate {learning_rate_2}")
plt.xlabel("Iterations")
plt.ylabel("Cost Function Value")
plt.title("Cost Function vs Iterations for Different Learning Rates")
plt.legend()
plt.show()

# Step 7: Confusion matrix and metrics (for question 5)
def confusion_matrix_and_metrics(X, Y, theta):
    predictions = sigmoid(np.dot(X, theta)) >= 0.5
    TP = np.sum((predictions == 1) & (Y == 1))
    TN = np.sum((predictions == 0) & (Y == 0))
    FP = np.sum((predictions == 1) & (Y == 0))
    FN = np.sum((predictions == 0) & (Y == 1))

    accuracy = (TP + TN) / len(Y)
    precision = TP / (TP + FP) if TP + FP > 0 else 0
    recall = TP / (TP + FN) if TP + FN > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

    return np.array([[TP, FP], [FN, TN]]), accuracy, precision, recall, f1_score

conf_matrix, accuracy, precision, recall, f1_score = confusion_matrix_and_metrics(X, Y, theta_1)

print("Confusion Matrix:\n", conf_matrix)
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1_score:.2f}")
