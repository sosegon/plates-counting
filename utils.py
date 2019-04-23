import numpy as np
from scipy.signal import find_peaks as fp

def extract_area(frame, x, y, width, height):

    return frame[y:y + height, x:x+width, :]


def find_peaks(points, threshold=0.5):

    # Find the peaks
    peaks, _ = fp(points, distance=15)

    # Filter the peaks based on threshold
    valid_peaks = np.argwhere(points[peaks] > threshold)

    return peaks[valid_peaks.ravel()]

def normalize(points):
    # Normalize the points
    min_ = np.min(points)
    max_ = np.max(points)

    points = points - min_
    points = points / (max_ - min_)

    return points