import numpy as np
import matplotlib.pyplot as plt
import pathlib
import pickle
source_path = str(pathlib.Path(__file__).parent.absolute())
with open(source_path + '/quaternion.pkl', 'rb') as f:
    quaternion_array = pickle.load(f)

def quat_conjugate(q, scalar_first=True):
    q_conj = q.copy()
    if scalar_first:
        q_conj[1:] = -q_conj[1:]
    else:
        q_conj[:-1] = -q_conj[:-1]
    return q_conj

def quat_product(q1, q2, scalar_first=True):
    if scalar_first:
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
    else:
        x1, y1, z1, w1 = q1
        x2, y2, z2, w2 = q2
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    if scalar_first:
        return np.array([w, x, y, z])
    else:
        return np.array([x, y, z, w])
    
def diff_quaternion(q_traj, scalar_first=True):
    q_arr = np.asarray(q_traj, dtype=float)
    if q_arr.ndim != 2 or q_arr.shape[1] != 4:
        raise ValueError("q_traj must be (N,4) array with [w,x,y,z] or [x,y,z,w]")

    q_aligned = q_arr.copy()
    for i in range(1, len(q_arr)):
        dot_product = np.dot(q_aligned[i - 1], q_aligned[i])
        if dot_product < 0:
            q_aligned[i] = -q_aligned[i]

    q_conj = np.array([quat_conjugate(q, scalar_first=scalar_first) for q in q_aligned])
    diff_quats = np.array([quat_product(q2, q1, scalar_first=scalar_first)
                           for q1, q2 in zip(q_conj[:-1], q_aligned[1:])])
    return diff_quats



diff_quat = diff_quaternion(quaternion_array, scalar_first=True)

plt.figure(figsize=(12, 4))
plt.plot(diff_quat, label=['w', 'x', 'y', 'z'])
plt.title('Quaternion Differences')
plt.legend()
plt.tight_layout()
plt.show()
