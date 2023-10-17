import numpy as np
from algebra import inner, change_sign, quaternion_power, quaternion_product, quaternion_divide
def slerp_with_power(q1, q2, tau):
    if inner(q1, q2) < 0:
        q1 = change_sign(q1)
    q_slerp = quaternion_product(quaternion_power(quaternion_divide(q2, q1), tau), q1)
    return q_slerp

def slerp(q1, q2, tau):
    if inner(q1, q2) < 0:
        q1 = change_sign(q1)
    theta = np.arccos(np.abs(inner(q1, q2)))
    q_slerp = np.zeros(4)

    q_slerp.x = (np.sin((1 - tau) * theta) * q1.x + np.sin(tau * theta) * q2.x) / np.sin(theta)
    q_slerp.y = (np.sin((1 - tau) * theta) * q1.y + np.sin(tau * theta) * q2.y) / np.sin(theta)
    q_slerp.z = (np.sin((1 - tau) * theta) * q1.z + np.sin(tau * theta) * q2.z) / np.sin(theta)
    q_slerp.w = (np.sin((1 - tau) * theta) * q1.w + np.sin(tau * theta) * q2.w) / np.sin(theta)
    return q_slerp

def slerp_sat(q1, q2, theta_max):
    if inner(q1, q2) < 0:
        q1 = change_sign(q1)
    theta = np.arccos(np.abs(inner(q1, q2)))
    q_slerp = np.copy(q1)
    if theta > theta_max:
        q_slerp.w = (np.sin(theta - theta_max) * q1.w + np.sin(theta_max) * q2.w) / np.sin(theta)
        q_slerp.x = (np.sin(theta - theta_max) * q1.x + np.sin(theta_max) * q2.x) / np.sin(theta)
        q_slerp.y = (np.sin(theta - theta_max) * q1.y + np.sin(theta_max) * q2.y) / np.sin(theta)
        q_slerp.z = (np.sin(theta - theta_max) * q1.z + np.sin(theta_max) * q2.z) / np.sin(theta)
    return q_slerp


def intrinsic(q1, q2):
    """Geodesic distance between rotations within the SO(3) manifold.

        This function is equivalent to

            min(
                np.absolute(np.log(q1 / q2)),
                np.absolute(np.log(q1 / -q2))
            )

        which is a measure of the "distance" between two rotations.  Note
        that no normalization is performed, which means that if q1 and/or
        q2 do not have unit norm, this is a more general type of distance.
        If they are normalized, the result of this function is half the
        angle through which vectors rotated by q1 would need to be rotated
        to lie on the same vectors rotated by q2.

        Parameters
        ----------
        q1, q2 : QuaternionicArray
            Quaternionic arrays to be measured

        See also
        --------
        quaternionic.distance.rotor.chordal
        quaternionic.distance.rotor.intrinsic
        quaternionic.distance.rotation.chordal

        """
    qtemp = np.empty(4)
    a = (q1.w - q2.w)**2 + (q1.x - q2.x)**2 + (q1.y - q2.y)**2 + (q1.z - q2.z)**2
    b = (q1.w + q2.w)**2 + (q1.x + q2.x)**2 + (q1.y + q2.y)**2 + (q1.z + q2.z)**2
    if b > a:
        qtemp = quaternion_divide(q1, q2)
    else:
        qtemp = quaternion_divide(q1, -q2)
    qtemp = quaternion_log(qtemp)
    qout = quaternion_absolute(qtemp)
    qout *= 2
    return qout