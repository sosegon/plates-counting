import cv2 as cv
import numpy as np
from utils import extract_area, find_peaks, normalize

class PlatesCounter:
    """
    A class to count the plates coming out a shoot in a manufacturing machine.

    The counting is done by measuring the lightness in the area of the shoot for
    every frame. The result is a 1-D signal. The peaks of that curve correspond
    to a plate coming out of the shoot.

    ...

    Attributes
    ----------
    x_start : int
        Horizontal position of the top-left corner of the shoot area.
    y_start : int
        Vertical position of the top-left corner of the shoot area.
    width : int
        Width of the shoot area.
    height : int
        Height of the shoot area.
    light_history : ndarray
        Array of average lightness of the shoot area for every frame.
    peaks : ndarray
        Array of positions for peaks of the average lightness.

    Methods
    -------
    init(frame)
        Initializes the average lightness array.
    process_frame(frame)
        Processes a frame to add its average lightness to history.
    calculate_plates()
        Calculates the frames where the average lightness is high.
    """
    def __init__(self, x_start, y_start, width, height):
        """
        Parameters
        ----------
        x_start : int
            Horizontal position of the top-left corner of the shoot area.
        y_start : int
            Vertical position of the top-left corner of the shoot area.
        width : int
            Width of the shoot area.
        height : int
            Height of the shoot area.
        """
        self.x_start = x_start
        self.y_start = y_start
        self.width = width
        self.height = height
        self.light_history = None
        self.peaks = None

    def init(self, frame):
        """
        Initializes the average lightness array.

        Parameters
        ----------
        frame : ndarray
            3-channel image
        """

        # Extract the shoot area.
        area = extract_area(frame, self.x_start, self.y_start,
            self.width, self.height)

        # Change to color space to get the lightness.
        area = cv.cvtColor(area, cv.COLOR_BGR2HLS)

        # Get the average lightness and start the history array.
        light_avg = np.mean(area[:, :, 1])
        self.light_history = np.array(light_avg)

    def process_frame(self, frame):
        """
        Processes a frame to add its average lightness to history.

        Parameters
        ----------
        frame : ndarray
            3-channel image.
        """
        if self.light_history is None:
            raise ValueError('Counter is not initialized')

        area = extract_area(frame, self.x_start, self.y_start,
            self.width, self.height)

        area = cv.cvtColor(area, cv.COLOR_BGR2HLS)

        light_avg = np.mean(area[:, :, 1])
        self.light_history = np.hstack((self.light_history, light_avg))

    def calculate_plates(self):
        """
        Calculates the frames where the average lightness is high.
        The result is set to the peaks attribute.
        """
        if self.light_history is None:
            raise ValueError('Counter is not initialized')

        # The limits for lightness in HLS are 0 and 255 by default
        self.light_history = normalize(self.light_history, 0, 255)
        self.peaks = find_peaks(self.light_history)
