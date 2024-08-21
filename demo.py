# Demo of optimizing coefficients for class separation using a combined objective function of inter-class distance and intra-class variance

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from tkinter import Tk, filedialog
from sklearn.preprocessing import LabelEncoder

def select_csv_file():
    """Opens a file picker to select a CSV file."""
    root = Tk()
    root.withdraw()  # Hide the main window
    file_path = filedialog.askopenfilename(title="Select CSV file", filetypes=[("CSV files", "*.csv")])
    root.destroy()  # Close the file picker
    return file_path

def load_csv(file_path):
    """Loads the CSV file into a pandas DataFrame."""
    return pd.read_csv(file_path)

def find_class_column(df):
    """Finds the column in the DataFrame that corresponds to the class labels."""
    class_column = None
    for column in df.columns:
        if column.lower() == 'class':
            class_column = column
            break
    if class_column is None:
        raise ValueError("No column named 'class' found in the CSV.")
    return class_column

def combined_objective_function(W, X, y, alpha=1.0, beta=1.0):
    """Objective function to maximize inter-class distance and minimize intra-class variance."""
    W = W.reshape((-1, 1))
    XW = X @ W
    
    # Calculate the class centroids in the projected space
    centroids = np.array([XW[y == i].mean() for i in np.unique(y)])
    
    # Inter-class distance: Minimize the negative of the minimum distance between centroids
    inter_class_distance = np.min([np.abs(centroids[i] - centroids[j]) for i in range(len(centroids)) for j in range(len(centroids)) if i != j])
    
    # Intra-class compactness: Sum of variances within each class
    intra_class_variance = np.sum([np.var(XW[y == i]) for i in np.unique(y)])
    
    # Combined objective: Maximize inter-class distance and minimize intra-class variance
    return -alpha * inter_class_distance + beta * intra_class_variance

def optimize_coefficients(X, y):
    """Optimizes the coefficients for class separation."""
    np.random.seed(0)
    W0 = np.random.rand(X.shape[1])
    result = minimize(combined_objective_function, W0, args=(X, y, 1.0, 1.0), constraints=[{'type': 'eq', 'fun': lambda W: np.linalg.norm(W) - 1}])
    return result.x

def project_and_plot(X, y, W_optimal, class_names):
    """Projects the data using the optimized coefficients and plots the result."""
    XW = X @ W_optimal.reshape((-1, 1))

    plt.figure(figsize=(10, 6))
    unique_classes = np.unique(y)
    colors = plt.cm.jet(np.linspace(0, 1, len(unique_classes)))

    for i, color in zip(unique_classes, colors):
        plt.scatter(XW[y == i], np.zeros_like(XW[y == i]) + i, color=color, label=class_names[i])

    plt.title('Projection of Dataset Using Combined Optimized Coefficients')
    plt.xlabel('Projected Value')
    plt.yticks(range(len(unique_classes)), class_names)
    plt.legend()
    plt.show()

def main():
    # Select the CSV file
    file_path = select_csv_file()
    if not file_path:
        print("No file selected. Exiting...")
        return
    
    # Load the CSV file
    df = load_csv(file_path)
    
    # Identify the class column and prepare the data
    class_column = find_class_column(df)
    y = df[class_column].values
    
    # Convert class labels to numeric values if they are not already integers
    if not np.issubdtype(y.dtype, np.number):
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)
        class_names = label_encoder.classes_
    else:
        class_names = np.unique(y)
    
    X = df.drop(columns=[class_column]).values
    
    # Optimize the coefficients
    W_optimal = optimize_coefficients(X, y)
    
    # Project the data and plot the result
    project_and_plot(X, y, W_optimal, class_names)

if __name__ == "__main__":
    main()
