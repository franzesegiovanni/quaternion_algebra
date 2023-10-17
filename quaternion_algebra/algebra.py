import numpy as np

def inner(q1, q2):
    result = q1.x*q2.x + q1.y*q2.y + q1.z*q2.z + q1.w*q2.w
    return result

def change_sign(q):
    q.w = -q.w
    q.x = -q.x
    q.y = -q.y
    q.z = -q.z
    return q

def quaternion_distance(q1, q2):
    if inner(q1, q2) < 0:
        q1 = change_sign(q1)
    theta = np.arccos(2 * inner(q1, q2) ** 2 - 1)
    return theta

def quaternion_conjugate(q):
    q.x = -q.x
    q.y = -q.y
    q.z = -q.z
    return q

def quaternion_power(q, s):
    b = np.sqrt(q.x**2 + q.y**2 + q.z**2)
    v = np.arctan2(b, q.w)
    f = v / b
    qout = np.copy(q)
    qout.w = np.log(q.w * q.w + b * b) / 2.0
    qout.x = f * q.x
    qout.y = f * q.y
    qout.z = f * q.z
    qout *= s
    vnorm = np.sqrt(qout.x**2 + qout.y**2 + qout.z**2)
    e = np.exp(qout.w)
    qout.w = e * np.cos(vnorm)
    qout.x *= e * np.sin(vnorm) / vnorm
    qout.y *= e * np.sin(vnorm) / vnorm
    qout.z *= e * np.sin(vnorm) / vnorm
    return qout

def quaternion_product(q1, q2):
    """Multiply quaternions q1*q2"""
    a = q1.w*q2.w - q1.x*q2.x - q1.y*q2.y - q1.z*q2.z
    b = q1.w*q2.x + q1.x*q2.w + q1.y*q2.z - q1.z*q2.y
    c = q1.w*q2.y - q1.x*q2.z + q1.y*q2.w + q1.z*q2.x
    d = q1.w*q2.z + q1.x*q2.y - q1.y*q2.x + q1.z*q2.w
    qout = np.copy(q1)
    qout.w = a
    qout.x = b
    qout.y = c
    qout.z = d
    return qout

def quaternion_diff(q1, q2):
    ''' this is solving q2-q1 '''
    if inner(q1, q2) < 0:
        q1 = change_sign(q1)
    q_diff = quaternion_product(q2, quaternion_conjugate(q1))
    return q_diff

def quaternion_divide(q1, q2):
    """Divide quaternions q1/q2 = q1 * q2.inverse"""
    if inner(q1, q2) < 0:
        q2 = change_sign(q2)
    q2norm = q2.w**2 + q2.x**2 + q2.y**2 + q2.z**2
    a = (q1.w*q2.w + q1.x*q2.x + q1.y*q2.y + q1.z*q2.z) / q2norm
    b = (-q1.w*q2.x + q1.x*q2.w - q1.y*q2.z + q1.z*q2.y) / q2norm
    c = (-q1.w*q2.y + q1.x*q2.z + q1.y*q2.w - q1.z*q2.x) / q2norm
    d = (-q1.w*q2.z - q1.x*q2.y + q1.y*q2.x + q1.z*q2.w) / q2norm
    qout = np.copy(q1)
    qout.w = a
    qout.x = b
    qout.y = c
    qout.z = d
    return qout

def quaternion_absolute(q):
    """Return absolute value of quaternion |q|"""
    qout = np.sqrt(q.w**2 + q.x**2 + q.y**2 + q.z**2)
    return qout

def quaternion_exp(q):
    """Return exponential of input quaternion exp(q)"""
    vnorm = np.sqrt(q.x**2 + q.y**2 + q.z**2)
    s = np.sin(vnorm) / vnorm
    e = np.exp(q.w)
    qout = np.copy(q)
    qout.w = e * np.cos(vnorm)
    qout.x = e * s * q.x
    qout.y = e * s * q.y
    qout.z = e * s * q.z
    return qout

def quaternion_log(q):
    """Return logarithm of input quaternion log(q)"""
    b = np.sqrt(q.x**2 + q.y**2 + q.z**2)
    v = np.arctan2(b, q.w)
    f = v / b
    qout = np.copy(q)
    qout.w = np.log(q.w * q.w + b * b) / 2.0
    qout.x = f * q.x
    qout.y = f * q.y
    qout.z = f * q.z
    return qout



