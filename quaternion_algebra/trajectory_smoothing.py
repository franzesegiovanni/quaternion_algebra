import numpy as np
import quaternion
def diff_quaternions(q_traj):
    # Convert to numpy quaternion array
    quaternion_array = np.array([quaternion.quaternion(q[0], q[1], q[2], q[3]) for q in q_traj])

    # Check dot product and flip quaternions if needed to ensure shorter geodesics
    quaternion_array_aligned = quaternion_array.copy()
    for i in range(1, len(quaternion_array)):
        dot_product = np.dot([quaternion_array[i-1].w, quaternion_array[i-1].x, quaternion_array[i-1].y, quaternion_array[i-1].z],
                            [quaternion_array[i].w, quaternion_array[i].x, quaternion_array[i].y, quaternion_array[i].z])
        if dot_product < 0:
            quaternion_array_aligned[i] = -quaternion_array[i]


    quaternion_array_conjugate = np.conjugate(quaternion_array_aligned)
    quaternion_difference = quaternion_array_aligned[1:] * quaternion_array_conjugate[:-1]

    quaternion_difference_array = np.array([[q.w, q.x, q.y, q.z] for q in quaternion_difference])

    #ensure continuity of the difference quaternions
    for i in range(1, len(quaternion_difference_array)):
        dot_product = np.dot(quaternion_difference_array[i-1], quaternion_difference_array[i])
        if dot_product < 0:
            quaternion_difference_array[i] = -quaternion_difference_array[i] # remember that q and -q represent the same rotation
    
    
    return quaternion_difference_array

