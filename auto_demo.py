import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from tkinter import Tk, filedialog

def select_folder():
    """Opens a file picker to select a folder containing CSV files, starting at the current working directory."""
    root = Tk()
    root.withdraw()  # Hide the main window
    current_dir = os.getcwd()  # Get the current working directory
    folder_path = filedialog.askdirectory(initialdir=current_dir, title="Select Folder with CSV files")
    root.destroy()  # Close the file picker
    return folder_path

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
    if len(centroids) < 2:
        inter_class_distance = 0
    else:
        inter_class_distance = np.min([np.abs(centroids[i] - centroids[j]) for i in range(len(centroids)) for j in range(len(centroids)) if i != j])
    
    intra_class_variance = np.sum([np.var(XW[y == i]) for i in np.unique(y)])
    
    return -alpha * inter_class_distance + beta * intra_class_variance

def optimize_coefficients(X, y):
    """Optimizes the coefficients for class separation."""
    np.random.seed(0)
    W0 = np.random.rand(X.shape[1])
    result = minimize(combined_objective_function, W0, args=(X, y, 1.0, 1.0), constraints=[{'type': 'eq', 'fun': lambda W: np.linalg.norm(W) - 1}])
    return result.x

def save_plot_and_stats(file_name, conf_matrix, accuracy, precision, recall, W_optimal, attribute_names, centroids, class_names, output_folder, X, y):
    """Saves the plot and statistics to the output folder."""
    plt.figure(figsize=(10, 4))
    
    XW = X @ W_optimal.reshape((-1, 1))
    sorted_centroids = sorted(centroids)
    unique_classes = np.unique(y)
    colors = plt.cm.jet(np.linspace(0, 1, len(unique_classes)))

    # Draw separation lines between nearest centroids (n-1 lines)
    for i in range(len(sorted_centroids) - 1):
        midpoint = (sorted_centroids[i] + sorted_centroids[i + 1]) / 2
        plt.axvline(x=midpoint, color='gray', linestyle=':', linewidth=1, label='Separation Line')

    # Scatter plot for each class
    predictions = np.array([unique_classes[np.argmin(np.abs(x - centroids))] for x in XW])
    misclassified = predictions != y

    for i, color in zip(unique_classes, colors):
        class_projection = XW[y == i]

        # Correctly classified points
        plt.scatter(class_projection[~misclassified[y == i]], 
                   np.zeros_like(class_projection[~misclassified[y == i]]) + i, 
                   color=color, label=class_names[i])

        # Misclassified points with the same class color but marked with 'x'
        plt.scatter(class_projection[misclassified[y == i]], 
                   np.zeros_like(class_projection[misclassified[y == i]]) + i, 
                   color=color, marker='x', s=100, linewidths=2, 
                   label=f'Misclassified {class_names[i]}')

        plt.axvline(x=class_projection.min(), color=color, linestyle='--', linewidth=1)
        plt.axvline(x=class_projection.max(), color=color, linestyle='--', linewidth=1)
        plt.scatter(centroids[i], i, color='black', marker='x', s=100, linewidths=2, label=f'Centroid of {class_names[i]}')

    plt.title('Projection of Dataset Using Combined Optimized Coefficients')
    plt.xlabel('Projected Value')
    plt.ylabel('Class')
    plt.yticks(range(len(unique_classes)), class_names)
    plt.legend()
    plt.savefig(os.path.join(output_folder, f'{file_name}_plot.png'))
    plt.close()

    with open(os.path.join(output_folder, f'{file_name}_stats.txt'), 'w') as f:
        f.write("Confusion Matrix:\n")
        f.write(np.array2string(conf_matrix) + "\n\n")
        f.write(f"Accuracy: {accuracy:.2%}\n")
        f.write(f"Precision: {precision:.2%}\n")
        f.write(f"Recall: {recall:.2%}\n")

        linear_combo = ' + '.join([f'{round(w, 4)}*{attr}' for w, attr in zip(W_optimal, attribute_names)])
        f.write(f"Linear Combination (W * X):\n{linear_combo}\n\n")

        # Write out the classifier ranges
        classifier_str = "Classifier:\n"
        sorted_centroids = sorted(centroids)
        
        for i in range(len(sorted_centroids)):
            if i == 0:
                classifier_str += f"If 0 <= {linear_combo} < {sorted_centroids[i]:.4f}, then class = {class_names[i]}\n"
            else:
                if i < len(sorted_centroids) - 1:
                    classifier_str += f"If {sorted_centroids[i-1]:.4f} <= {linear_combo} < {sorted_centroids[i]:.4f}, then class = {class_names[i]}\n"
                else:
                    classifier_str += f"If {sorted_centroids[i-1]:.4f} <= {linear_combo}, then class = {class_names[i]}\n"

        f.write(f"\n{classifier_str}")

def process_all_csvs_in_folder(folder_path, output_folder):
    """Processes all CSV files in the selected folder and saves results to the output folder."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    summary_stats = []

    for file_name in os.listdir(folder_path):
        if file_name.endswith('.csv'):
            file_path = os.path.join(folder_path, file_name)
            df = load_csv(file_path)

            class_column = find_class_column(df)
            y = df[class_column].values
            attribute_names = df.drop(columns=[class_column]).columns

            if not np.issubdtype(y.dtype, np.number):
                label_encoder = LabelEncoder()
                y = label_encoder.fit_transform(y)
                class_names = label_encoder.classes_
            else:
                class_names = np.unique(y)

            X = df.drop(columns=[class_column]).values
            W_optimal = optimize_coefficients(X, y)
            XW = X @ W_optimal.reshape((-1, 1))

            centroids = np.array([XW[y == i].mean() for i in np.unique(y)])
            predictions = np.array([np.argmin(np.abs(x - centroids)) for x in XW])
            conf_matrix = confusion_matrix(y, predictions, labels=np.unique(y))
            
            # Handle cases where a class has no predicted samples
            precision = precision_score(y, predictions, labels=np.unique(y), average='weighted', zero_division=0)
            recall = recall_score(y, predictions, labels=np.unique(y), average='weighted', zero_division=0)

            accuracy = np.trace(conf_matrix) / np.sum(conf_matrix)

            save_plot_and_stats(file_name, conf_matrix, accuracy, precision, recall, W_optimal, attribute_names, centroids, class_names, output_folder, X, y)

            summary_stats.append([file_name, f"{accuracy:.2%}", f"{precision:.2%}", f"{recall:.2%}"])

    # Sort the summary stats by file name
    summary_stats = sorted(summary_stats, key=lambda x: x[0])

    # Save summary of all CSVs
    summary_df = pd.DataFrame(summary_stats, columns=['File Name', 'Accuracy', 'Precision', 'Recall'])
    summary_df.to_csv(os.path.join(output_folder, 'summary_stats.csv'), index=False)

def main():
    folder_path = select_folder()
    output_folder = "output_results"
    
    if not folder_path:
        print("No folder selected. Exiting...")
        return

    process_all_csvs_in_folder(folder_path, output_folder)
    print(f"Processing complete. Results saved to {output_folder}.")

if __name__ == "__main__":
    main()
