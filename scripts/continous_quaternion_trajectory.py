import numpy as np
import matplotlib.pyplot as plt
import pathlib
import pickle
import quaternion
from quaternion_algebra.algebra import to_euler_angles
from quaternion_algebra.trajectory_smoothing import diff_quaternions
source_path = str(pathlib.Path(__file__).parent.absolute())
with open(source_path + '/quaternion.pkl', 'rb') as f:
    quaternion_array = pickle.load(f)

quaternion_list = np.array([quaternion.quaternion(q[0], q[1], q[2], q[3]) for q in quaternion_array])
quaternion_difference = diff_quaternions(quaternion_array)
# Reconstruct the original quaternion trajectory by integration
continuous_quaternions = [quaternion_list[0]]  # Start with the first quaternion

for i in range(len(quaternion_difference)):
    # Convert difference back to quaternion object
    diff_quat = quaternion.quaternion(quaternion_difference[i][0], 
                                    quaternion_difference[i][1], 
                                    quaternion_difference[i][2], 
                                    quaternion_difference[i][3])

    # Integrate: q_next = q_current * diff_quat
    next_quat = diff_quat * continuous_quaternions[-1] 
    continuous_quaternions.append(next_quat)

# continuous_quaternions = np.array(continuous_quaternions)

# Convert back to array format for comparison
continuous_array = np.array([[q.w, q.x, q.y, q.z] for q in continuous_quaternions])
# Convert original quaternions to Euler angles
original_euler = np.array([to_euler_angles(q) for q in quaternion_list])
continuous_euler = np.array([to_euler_angles(q) for q in continuous_quaternions])

plt.figure()
plt.plot(quaternion_difference)
plt.title('Quaternion Differences')
plt.legend(['w', 'x', 'y', 'z'])

# Plot comparison
plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
plt.plot(quaternion_array, label=['w', 'x', 'y', 'z'])
plt.title('Original Quaternion Trajectory')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(continuous_array, label=['w', 'x', 'y', 'z'])
plt.title('continuous Quaternion Trajectory')
plt.legend()

plt.tight_layout()


# Plot Euler angles comparison
plt.figure(figsize=(12, 10))

plt.subplot(3, 1, 1)
plt.plot(original_euler[:, 0], 'b-', label='Original Roll')
plt.plot(continuous_euler[:, 0], 'r--', label='continuous Roll')
plt.title('Roll Angle Comparison')
plt.ylabel('Angle (radians)')
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(original_euler[:, 1], 'b-', label='Original Pitch')
plt.plot(continuous_euler[:, 1], 'r--', label='continuous Pitch')
plt.title('Pitch Angle Comparison')
plt.ylabel('Angle (radians)')
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(original_euler[:, 2], 'b-', label='Original Yaw')
plt.plot(continuous_euler[:, 2], 'r--', label='continuous Yaw')
plt.title('Yaw Angle Comparison')
plt.ylabel('Angle (radians)')
plt.xlabel('Time Step')
plt.legend()

plt.tight_layout()
plt.show()
