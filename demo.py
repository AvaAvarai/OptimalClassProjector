import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.optimize import minimize
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from tkinter import Tk, filedialog, Frame, Label, TOP, BOTH, BOTTOM

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
    for column in df.columns:
        if column.lower() == 'class':
            return column
    raise ValueError("No column named 'class' found in the CSV.")

def combined_objective_function(W, X, y, alpha=1.0, beta=1.0):
    """Objective function to maximize inter-class distance and minimize intra-class variance."""
    W = W.reshape((-1, 1))
    XW = X @ W
    
    centroids = np.array([XW[y == i].mean() for i in np.unique(y)])
    inter_class_distance = np.min([np.abs(centroids[i] - centroids[j]) for i in range(len(centroids)) for j in range(len(centroids)) if i != j])
    intra_class_variance = np.sum([np.var(XW[y == i]) for i in np.unique(y)])
    
    return -alpha * inter_class_distance + beta * intra_class_variance

def optimize_coefficients(X, y):
    """Optimizes the coefficients for class separation."""
    np.random.seed(0)
    W0 = np.random.rand(X.shape[1])
    result = minimize(combined_objective_function, W0, args=(X, y, 1.0, 1.0), constraints=[{'type': 'eq', 'fun': lambda W: np.linalg.norm(W) - 1}])
    return result.x

def display_confusion_matrix_and_stats(conf_matrix, accuracy, precision, recall, W_optimal, attribute_names, centroids, class_names, root):
    """Displays the confusion matrix, stats, coefficients, and classifier in a Tkinter window."""
    stats_frame = Frame(root)
    stats_frame.pack(side=BOTTOM, fill=BOTH, expand=True)

    matrix_str = "Confusion Matrix:\n"
    for row in conf_matrix:
        matrix_str += " ".join(f"{val:4d}" for val in row) + "\n"

    # Construct the linear combination string
    linear_combo = " + ".join([f"{round(coef, 4)}*{attr}" for coef, attr in zip(W_optimal.flatten(), attribute_names)])

    # Define the classifier based on separation lines
    classifier_str = "Classifier:\n"
    sorted_centroids = sorted(centroids)
    
    # First class: from 0 up to the first separation line
    classifier_str += f"If 0 <= {linear_combo} < {sorted_centroids[0]:.4f}, then class = {class_names[0]}\n"

    # Middle classes: for each middle centroid, define a range between previous and current
    for i in range(1, len(sorted_centroids)):
        if i < len(sorted_centroids) - 1:
            classifier_str += f"If {sorted_centroids[i-1]:.4f} <= {linear_combo} < {sorted_centroids[i]:.4f}, then class = {class_names[i]}\n"
        else:
            # Last class: everything greater than or equal to the last separation line
            classifier_str += f"If {sorted_centroids[i-1]:.4f} <= {linear_combo}, then class = {class_names[i]}\n"

    stats_str = (
        f"\nAccuracy: {accuracy:.4f} = {accuracy * 100:.2f}%\n"
        f"Precision: {precision:.4f} = {precision * 100:.2f}%\n"
        f"Recall: {recall:.4f} = {recall * 100:.2f}%\n"
        f"Linear Combination (W * X):\n{linear_combo}\n"
        f"\n{classifier_str}"
    )

    Label(stats_frame, text=matrix_str, justify='left', font=("Helvetica", 12)).pack(side=TOP, anchor="w")
    Label(stats_frame, text=stats_str, justify='left', font=("Helvetica", 12)).pack(side=TOP, anchor="w")

def project_and_plot_tkinter(X, y, W_optimal, class_names, attribute_names, root):
    """Projects the data using the optimized coefficients and plots the result with subspace boundaries and centroids in a Tkinter window."""
    XW = X @ W_optimal.reshape((-1, 1))

    plot_frame = Frame(root)
    plot_frame.pack(side=TOP, fill=BOTH, expand=False)

    fig, ax = plt.subplots(figsize=(10, 4))  # Reduced height
    unique_classes = np.unique(y)
    colors = plt.cm.jet(np.linspace(0, 1, len(unique_classes)))

    centroids = np.array([XW[y == i].mean() for i in unique_classes])
    sorted_centroids = sorted(centroids)

    # Draw separation lines between nearest centroids (n-1 lines)
    for i in range(len(sorted_centroids) - 1):
        midpoint = (sorted_centroids[i] + sorted_centroids[i + 1]) / 2
        ax.axvline(x=midpoint, color='gray', linestyle=':', linewidth=1, label='Separation Line')

    # Predictions and identification of misclassified points
    predictions = np.array([unique_classes[np.argmin(np.abs(x - centroids))] for x in XW])
    misclassified = predictions != y

    for i, color in zip(unique_classes, colors):
        class_projection = XW[y == i]

        # Correctly classified points
        ax.scatter(class_projection[~misclassified[y == i]], 
                   np.zeros_like(class_projection[~misclassified[y == i]]) + i, 
                   color=color, label=class_names[i])

        # Misclassified points with the same class color but marked with 'x'
        ax.scatter(class_projection[misclassified[y == i]], 
                   np.zeros_like(class_projection[misclassified[y == i]]) + i, 
                   color=color, marker='x', s=100, linewidths=2, 
                   label=f'Misclassified {class_names[i]}')

        ax.axvline(x=class_projection.min(), color=color, linestyle='--', linewidth=1)
        ax.axvline(x=class_projection.max(), color=color, linestyle='--', linewidth=1)
        ax.scatter(centroids[i], i, color='black', marker='x', s=100, linewidths=2, label=f'Centroid of {class_names[i]}')

    ax.set_title('Projection of Dataset Using Combined Optimized Coefficients')
    ax.set_xlabel('Projected Value')
    ax.set_yticks(range(len(unique_classes)))
    ax.set_yticklabels(class_names)
    ax.legend()

    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=True)

    conf_matrix = confusion_matrix(y, predictions, labels=unique_classes)
    accuracy = np.trace(conf_matrix) / np.sum(conf_matrix)
    precision = precision_score(y, predictions, labels=unique_classes, average='weighted')
    recall = recall_score(y, predictions, labels=unique_classes, average='weighted')

    display_confusion_matrix_and_stats(conf_matrix, accuracy, precision, recall, W_optimal, attribute_names, centroids, class_names, root)

def main():
    file_path = select_csv_file()
    if not file_path:
        print("No file selected. Exiting...")
        return
    
    df = load_csv(file_path)
    class_column = find_class_column(df)
    y = df[class_column].values
    attribute_names = df.drop(columns=[class_column]).columns  # Get attribute names
    
    if not np.issubdtype(y.dtype, np.number):
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)
        class_names = label_encoder.classes_
    else:
        class_names = np.unique(y)
    
    X = df.drop(columns=[class_column]).values
    W_optimal = optimize_coefficients(X, y)
    
    root = Tk()
    root.title("Optimal Class Projector")
    project_and_plot_tkinter(X, y, W_optimal, class_names, attribute_names, root)
    root.mainloop()

if __name__ == "__main__":
    main()
