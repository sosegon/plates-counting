import cv2 as cv
import numpy as np
from utils import extract_area, find_peaks, normalize

class PlatesCounter:

    def __init__(self, x_start, y_start, width, height):
        self.x_start = x_start
        self.y_start = y_start
        self.width = width
        self.height = height
        self.light_history = None
        self.peaks = None

    def init(self, frame):
        area = extract_area(frame, self.x_start, self.y_start,
            self.width, self.height)

        area = cv.cvtColor(area, cv.COLOR_BGR2HLS)

        light_avg = np.mean(area[:, :, 1])
        self.light_history = np.array(light_avg)

    def process_frame(self, frame):
        if self.light_history is None:
            raise ValueError('Counter is not initialized')

        area = extract_area(frame, self.x_start, self.y_start,
            self.width, self.height)

        area = cv.cvtColor(area, cv.COLOR_BGR2HLS)

        light_avg = np.mean(area[:, :, 1])
        self.light_history = np.hstack((self.light_history, light_avg))

    def calculate_plates(self):
        self.light_history = normalize(self.light_history)
        self.peaks = find_peaks(self.light_history)
