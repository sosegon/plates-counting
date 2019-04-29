import argparse
import numpy as np
from scipy.signal import find_peaks as fp
import time
import platform

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
    """
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
    return (points - min_) / (max_ - min_)

def readable_time(seconds):
    """
    Converts time in seconds to human-readable format.

    Parameters
    ----------
    seconds : float
        Time to convert.

    Returns
    -------
    str : Time in HH:MM:SS format.
    """
    return time.strftime('%H:%M:%S', time.gmtime(seconds))

def time_and_date():
    """
    Gets the current time in human-readable format.

    Returns
    -------
    str : Time in YYYY_MM_DD_HH_MM format.
    """
    return time.strftime('%Y_%m_%d_%H_%M')

def get_file_simple_name(file_name):
    """
    Gets the simple name of a file. It removes the path part.

    Parameters
    ----------
    file_name : str
        Path to the file

    Returns
    -------
    str : Simple name of the file.
    """
    if 'Linux' in platform.system():
        if file_name.endswith('/'):
            return file_name.split('/')[-2]
        else:
            return file_name.split('/')[-1]

    elif 'Windows' in platform.system():
        if file_name.endswith('\\'):
            return file_name.split('\\')[-2]
        else:
            return file_name.split('\\')[-1]

def string_to_bool(value):
    """
    Converts a string to an the equivalent bool value.

    Parameters
    ----------
    value : str
        Value to convert.

    Returns
    -------
        bool : Equivalent bool value.
    """
    if value.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')