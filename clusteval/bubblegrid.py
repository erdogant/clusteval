import matplotlib.pyplot as plt
import numpy as np

def plot_bubble_grid(df, bubble_size, color_map='Blues'):
    """
    Function to plot a bubble grid chart with a matrix as input.

    Parameters:
    -- matrix: numpy array
        Input matrix to plot as a bubble grid chart.
    -- bubble_size: int
        Bubble size for each grid cell.
    -- color_map: str, optional (default: 'Blues')
        Color map for the bubble grid chart.

    Returns:
    -- None
    """
    rows, cols = df.shape
    x = np.arange(cols)
    y = np.arange(rows)
    X, Y = np.meshgrid(x, y)
    Z = df.values
    bubble_sizes = bubble_size * Z.flatten()  # Flatten the matrix and scale the bubble sizes

    # Plot the bubble grid chart
    plt.scatter(X.flatten(), Y.flatten(), s=bubble_sizes, c=Z.flatten(), cmap=color_map)
    plt.colorbar(label='Value')
    plt.xticks(df.columns)
    plt.yticks(df.index)
    plt.ylabel('Y Axis')
    plt.title('Bubble Grid Chart')
    plt.show()


# %%

import numpy as np
import pandas as pd

# Input array
arr = np.array([[3, 2, 1, 1, 1, 3],
                [2, 2, 1, 1, 1, 2],
                [4, 3, 1, 1, 1, 4],
                [4, 3, 1, 1, 1, 5],
                [5, 4, 1, 1, 2, 7]], dtype=np.int32)

# Get the number of columns in the input array
num_cols = arr.shape[1]

# Initialize a dictionary to store the counts for each element
occurrence_dict = {}

# Loop through each column and count the occurrences for each element
for i in range(num_cols):
    col = arr[:, i]
    unique_elements, counts = np.unique(col, return_counts=True)
    for e, count in zip(unique_elements, counts):
        if e in occurrence_dict:
            occurrence_dict[e].append(count)
        else:
            occurrence_dict[e] = [count]

# Fill in missing counts with zeros
for key in occurrence_dict.keys():
    if len(occurrence_dict[key]) < num_cols:
        occurrence_dict[key].extend([0] * (num_cols - len(occurrence_dict[key])))

# Convert the dictionary to a pandas DataFrame
df = pd.DataFrame(occurrence_dict)

print("Input array:")
print(arr)
print("Occurrence DataFrame:")
print(df)
df = df.T


# Bubble size for each grid cell
bubble_size = 100
# Color map for the bubble grid chart
color_map = 'cool'

plot_bubble_grid(df, bubble_size, color_map)

