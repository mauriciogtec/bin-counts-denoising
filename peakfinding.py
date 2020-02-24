import numpy as np


def findpeaks(phat: np.ndarray, normalize=True, minval=0.1):
    C = phat.max() if normalize else 1.0
    clusters = []
    current = []
    for i, x in enumerate(phat):
        if x > C * minval:
            current.append(i)
        elif len(current) > 0:
            clusters.append(current)
            current = []
    peaks = [np.argmax(phat[cl]) for cl in clusters]
    return peaks
