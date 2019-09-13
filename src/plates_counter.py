import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from utils import extract_area, find_peaks, normalize, readable_time
from section_processor import SectionProcessor

class PlatesCounter(SectionProcessor):
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
    events : ndarray
        Positions where the state changes.

    Methods
    -------
    init(frame)
        Initializes the average lightness array.
    process_frame(frame)
        Processes a frame to add its average lightness to history.
    calculate_position()
        Calculates the frames where the average lightness is high.
    generate_report(fps, sub_name)
        Generates a report with the timestaps of the plates coming out the
        shoot.
    draw_caption(value, frame, font, position=(0, 0), color=(0, 0, 255))
        Draws the number of plates in a frame.
    draw_processing_info(frame_number, frame, font, position=(0, 0), color=(0, 0, 255))
        Draws the number of plates in a frame.
    show_processing(frame, name="Plates")
        Displays the lightness of the shoot.
    plot(name="Plates")
        Plots the lightness of the shoot in every frame. The 'X's correspond to
        plates.
    calculate_events(fps, last_frame)
        Calculates the points where the state changes.
    state_at_frame(frame_number)
        Returns the state of the processor at a given frame.
    extract_section(frame)
        Extracts the section of the frame where the processor works.
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
        self.section_width = width
        self.section_height = height
        self.light_history = None
        self.peaks = None
        self.events = np.array([])

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
            self.section_width, self.section_height)

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
            self.section_width, self.section_height)

        area = cv.cvtColor(area, cv.COLOR_BGR2HLS)

        light_avg = np.mean(area[:, :, 1])
        self.light_history = np.hstack((self.light_history, light_avg))

    def calculate_positions(self):
        """
        Calculates the frames where the average lightness is high.
        The result is set to the peaks attribute.
        """
        self.__calculate_plates()

    def __calculate_plates(self):
        """
        Calculates the frames where the average lightness is high.
        The result is set to the peaks attribute.
        Method just for easy reading.
        """
        if self.light_history is None:
            raise ValueError('Counter is not initialized')

        # The limits for lightness in HLS are 0 and 255 by default
        self.light_history = normalize(self.light_history, 0, 255)
        self.peaks = find_peaks(self.light_history)

    def generate_report(self, fps, sub_name):
        """
        Generates a report with the timestaps of the plates coming out the
        shoot.
        The report is written to a file named plates_$sub_name$.csv

        Parameters
        ----------
        fps : float
            Number of frames per second of the processed video.
        sub_name : str
            Partial name of the file report.
        """
        timestamps = np.copy(self.peaks) / fps
        file_name = 'plates_{}.csv'.format(sub_name)

        if timestamps.shape[0] > 0:
            func = np.vectorize(readable_time)
            array_to_dump = func(timestamps)
        else:
            array_to_dump = np.array(['No plates'])

        np.savetxt(file_name, array_to_dump, delimiter=',', fmt='%s')

    def draw_caption(self, value, frame, font, position=(0, 0), color=(0, 0, 255)):
        """
        Draws the number of plates in a frame.

        Parameters
        ----------
        value : int
            Number of plates.
        frame : ndarray
            3-channel image.
        font : int
            Font type available in OpenCV.
        position : (int, int)
            Position in the frame to start drawing the caption.
        color : (int, int, int)
            BGR color of the caption.
        """
        text = 'Plates: {}'.format(value)
        super().draw_text(text, frame, font, color, (self.x_start, self.y_start))

    def draw_processing_info(self, frame_number, frame, font, position=(0, 0), color=(0, 0, 255)):
        """
        Draws the number of plates in a frame.

        Parameters
        ----------
        frame_number : int
            Value to compare to internal information of processor.
        frame : ndarray
            3-channel image.
        font : int
            Font type available in OpenCV.
        position : (int, int)
            Position in the frame to start drawing the number of plates.
        color : (int, int, int)
            BGR color of the number of plates.
        """
        valid_frames = self.peaks[self.peaks <= frame_number]

        self.draw_caption(valid_frames.shape[0], frame, font, position, color)

    def show_processing(self, frame, name="Plates"):
        """
        Displays the lightness of the shoot.

        Parameters
        ----------
        frame : ndarray
            3-channel image.
        name : str
            Name of the window to display the frame.
        """
        shoot = extract_area(frame, self.x_start, self.y_start,
            self.section_width, self.section_height)

        H = shoot.shape[0]
        W = shoot.shape[1]

        shoot = cv.cvtColor(shoot, cv.COLOR_BGR2HLS)

        # Resize for better visualization
        shoot = cv.resize(shoot, (W * 3, H * 3))

        cv.imshow(name, shoot[:,:,1])

    def plot(self, name="Plates"):
        """
        Plots the lightness of the shoot in every frame. The 'X's correspond to
        plates.

        Parameters
        ----------
        name : str
            Name of the plot.
        """
        plt.plot(self.light_history)
        plt.plot(self.peaks, self.light_history[self.peaks], 'X')
        plt.savefig(name)
        plt.figure()

    def calculate_events(self, fps, last_frame):
        """
        Calculate the points (frame indices) where the PlatesCounter changes its
        state. The state is defined by the number of plates.

        Parameters
        ----------
        fps : float
            Frames per second of the processed video.
        last_frame : int
            Index of the last frame in the processed video
        """
        if self.peaks.shape[0] > 0:
            self.events = np.ones((self.peaks.shape[0], 2))
            self.events[:,0] = self.peaks
            self.events[:,1] = np.arange(self.peaks.shape[0]) + 1


            self.events = np.vstack(([0, 0], self.events))
            self.events = np.vstack((self.events, [last_frame, self.events[-1, 1]]))
            self.events = self.events.astype(int)

    def state_at_frame(self, frame_number):
        """
        Returns the state at a given frame.

        Parameters
        ----------
        frame_number : int
            The point to get the state.

        Returns
        -------
        int : The state at a given frame.
        """
        indices = np.argwhere(self.events[:, 0] <= frame_number).ravel()
        state = None
        if indices.shape[0] > 0:
            state = self.events[indices[-1], 1]

        return state

    def extract_section(self, frame):
        """
        Extracts the section of the frame where the processor works.

        Parameters
        ----------
        frame : ndarray
            3-channel image.

        Returns
        -------
        ndarray : 3-channeld section of the frame.
        """
        return extract_area(frame, self.x_start, self.y_start,
            self.section_width, self.section_height)
