import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import Tk, filedialog, Frame, TOP, BOTH
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
    for column in df.columns:
        if column.lower() == 'class':
            return column
    raise ValueError("No column named 'class' found in the CSV.")

def project_and_plot_tkinter(X, y, class_names, attribute_names, root):
    """Plots the data in a 3D cube as parallel coordinates."""
    num_attributes = X.shape[1]

    plot_frame = Frame(root)
    plot_frame.pack(side=TOP, fill=BOTH, expand=False)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    unique_classes = np.unique(y)
    colors = plt.cm.jet(np.linspace(0, 1, len(unique_classes)))

    # Scatter plots and connect the points to form parallel coordinates
    for class_index, class_value in enumerate(unique_classes):
        z_plane = class_index  # Each class gets its own z-plane
        class_mask = y == class_value
        class_color = colors[class_index]

        # Scatter plot on the z_plane
        for i in range(num_attributes):
            ax.scatter(X[class_mask, i], np.full(X[class_mask, i].shape, i), z_plane, color=class_color)

        # Connect points across axes on the z-plane
        for sample_index in np.where(class_mask)[0]:
            ax.plot(X[sample_index, :], np.arange(num_attributes), np.full(num_attributes, z_plane), color=class_color, alpha=0.5)

    ax.set_xlabel('Attribute Values')
    ax.set_ylabel('Attributes')
    ax.set_zlabel('Class Planes')
    ax.set_yticks(range(len(attribute_names)))
    ax.set_yticklabels(attribute_names)
    ax.set_zticks(range(len(unique_classes)))
    ax.set_zticklabels(class_names)
    ax.set_title('3D Parallel Coordinates Visualization')

    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=True)

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
    
    root = Tk()
    root.title("3D Parallel Coordinates")
    project_and_plot_tkinter(X, y, class_names, attribute_names, root)
    root.mainloop()

if __name__ == "__main__":
    main()
