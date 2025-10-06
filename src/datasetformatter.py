import pandas as pd
import numpy as np

  # Load the CSV file
df = pd.read_csv('PEMS-BAY.csv')

  # Convert the DataFrame to a NumPy array
data = df.to_numpy()

  # Save the array in NPZ format
np.savez_compressed('PEMS-BAY.npz', data=data)
