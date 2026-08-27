# test_setup.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print("✅ NumPy version:", np.__version__)
print("✅ Pandas version:", pd.__version__)
print("✅ Matplotlib version:", plt.matplotlib.__version__)

# Test simple array
arr = np.array([1, 2, 3, 4, 5])
print("✅ NumPy array:", arr)

# Test pandas
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
print("✅ Pandas DataFrame:")
print(df)

print("\n✅ All imports successful!")