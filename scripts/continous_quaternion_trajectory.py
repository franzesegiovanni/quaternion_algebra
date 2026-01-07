import numpy as np
import matplotlib.pyplot as plt
import pathlib
import pickle
source_path = str(pathlib.Path(__file__).parent.absolute())
with open(source_path + '/quaternion.pkl', 'rb') as f:
    quaternion_array = pickle.load(f)

def continuous_quaternion(quaternion_array):
    continuous_quaternions = quaternion_array.copy()
    for i in range(len(continuous_quaternions)-1):
        if np.dot(continuous_quaternions[i+1], continuous_quaternions[i]) < 0:
            continuous_quaternions[i+1] = -continuous_quaternions[i+1]
    return continuous_quaternions

continuous_quaternions = continuous_quaternion(quaternion_array)

# Plot comparison
plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
plt.plot(quaternion_array, label=['w', 'x', 'y', 'z'])
plt.title('Original Quaternion Trajectory')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(continuous_quaternions, label=['w', 'x', 'y', 'z'])
plt.title('continuous Quaternion Trajectory')
plt.legend()

plt.show()
