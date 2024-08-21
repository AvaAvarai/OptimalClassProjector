import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from tkinter import Tk, filedialog

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
    """Projects the data using the optimized coefficients and plots the result with subspace boundaries."""
    XW = X @ W_optimal.reshape((-1, 1))

    plt.figure(figsize=(10, 6))
    unique_classes = np.unique(y)
    colors = plt.cm.jet(np.linspace(0, 1, len(unique_classes)))

    for i, color in zip(unique_classes, colors):
        class_projection = XW[y == i]
        plt.scatter(class_projection, np.zeros_like(class_projection) + i, color=color, label=class_names[i])
        
        # Draw lines at the beginning and end of each class's subspace
        plt.axvline(x=class_projection.min(), color=color, linestyle='--', linewidth=1)
        plt.axvline(x=class_projection.max(), color=color, linestyle='--', linewidth=1)

    plt.title('Projection of Dataset Using Combined Optimized Coefficients')
    plt.xlabel('Projected Value')
    plt.yticks(range(len(unique_classes)), class_names)
    plt.legend()
    plt.show()

    # Predict the class by finding the closest centroid
    centroids = np.array([XW[y == i].mean() for i in unique_classes])
    predictions = np.array([unique_classes[np.argmin(np.abs(x - centroids))] for x in XW])
    
    # Calculate and print the confusion matrix
    conf_matrix = confusion_matrix(y, predictions, labels=unique_classes)
    accuracy = np.trace(conf_matrix) / np.sum(conf_matrix)
    
    print("Confusion Matrix:")
    print(conf_matrix)
    print(f"Accuracy: {accuracy:.4f} = {accuracy * 100:.2f}%")

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
