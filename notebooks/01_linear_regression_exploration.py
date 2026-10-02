import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print("Linear Regression From Scratch")

# Simple dataset
data = {
    "Hours": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Score": [35, 40, 48, 52, 60, 65, 72, 78, 85, 92]
}

df = pd.DataFrame(data)

print(df)
# Visualize the data
plt.scatter(df["Hours"], df["Score"])

plt.xlabel("Hours Studied")
plt.ylabel("Exam Score")
plt.title("Hours Studied vs Exam Score")

plt.show()