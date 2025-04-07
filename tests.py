# module for unit tests.
import numpy as np
# check that the whip/transmitter vectors are orthogonal to the magnetic field vectors
def check_whip_orthogonality(mag_vecs, whip_vecs):
    if mag_vecs.shape != whip_vecs.shape:
        raise ValueError("mag_vecs and whip_vecs must have the same shape")
    if mag_vecs.ndim != 2 or whip_vecs.ndim != 2:
        raise ValueError("mag_vecs and whip_vecs must be 2D arrays")
    dot_products = np.einsum('ij,ij->i',mag_vecs,whip_vecs)
    return np.all(dot_products<1e-5)
