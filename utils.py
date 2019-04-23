import numpy as np
from scipy.signal import find_peaks as fp

def extract_area(frame, x, y, width, height):
    """
    Extracts an area from a frame.

    Parameters
    ----------
    frame : ndarray
        3-channel image.
    x : int
        Horizonal position of top-left corner of the area.
    y : int
        Vertical position of top-left corner of the area.
    width : int
        Width of the area.
    height : int
        Height of the area.

    Returns
    -------
    ndarray : 3-channel image.
    """
    return frame[y:y + height, x:x+width, :]

def find_peaks(points, threshold=0.5):
    """
    Finds the peaks in a signal.

    Parameters
    ----------
    points : ndarray
        1-D array of normalized data points.
    threshold : float
        Limit to filter peaks; those below that value are discarded.

    Returns
    -------
    ndarray : 1-D array with the indices corresponding to the peaks of
        points.
    """

    # Find the peaks
    peaks, _ = fp(points, distance=15)

    # Filter the peaks based on threshold
    valid_peaks = np.argwhere(points[peaks] > threshold)

    return peaks[valid_peaks.ravel()]

def normalize(points, min_, max_):
    """`
    Normalizes the points of a signal.

    Parameters
    ----------
    points : ndarray
        1-D array of data points
    min_ : int
        Low limit of the range to normalize the points.
    max_ : int
        High limit of the range to normalize the points.

    Returns
    -------
    ndarray : 1-D array of normalized points.
    """
    return points - min_ / (max_ - min_)
