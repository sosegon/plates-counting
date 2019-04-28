import cv2 as cv
import numpy as np
from utils import normalize, find_peaks, extract_area, readable_time
from section_processor import SectionProcessor

class PressCounter(SectionProcessor):
    """
    A class to count press moves from a video frames of a
    manufacturing machine.

    The processing is done by tracking the press. To do so, it is required
    to define two areas. The main area corresponds to a section of the
    machine where the tracking will be performed. The second area (inner area)
    is contained within the main one. This area corresponds to the object that
    will be tracked.

    The tracking is performed with a built-in object tracker in OpenCV.

    During the tracking, the position of the top-left corner of the bounding
    box of the inner area is recorded in every frame. This process creates a set
    of points. Those points form a sine-shape curve. The peaks of that curve
    correspond to the lower position of the press. The number of peaks
    corresponds to the number of press moves.

    ...

    Attributes
    ----------
    x_start : int
        Horizonal position of top-left corner of the main area.
    y_start : int
        Vertical position of top-left corner of the main area.
    x_end : int
        Horizonal position of bottom-right corner of the main area.
    y_end : int
        Vertical position of bottom-right corner of the main area.
    x_bar_start : int
        Horizontal position of the top-left corner of the inner area. This
        position is relative to the main area.
    y_bar_start : int
        Vertical position of the top-left corner of the inner area. This
        position is relative to the main area.
    bbox : (int, int, int, int)
        Inner area define by the position of the top-left corner, width and
        height.
    y_pos_history : list
        Positions of the top-left corner of the inner area.
    tracker : cv2.Tracker
        Tracker that finds the position of the inner area.
    peaks : list
        Indices of the frames where the inner area is in the bottom position.

    Methods
    -------
    init_tracker(frame)
        Initializes the tracker.
    process_frame(frame)
        Processes the frame to update the inner area.
    calculate_press_down_positions()
        Calculates the frames where the inner area is in the bottom position.
    draw_inner_area(frame)
        Draws the inner area.
    """

    def __init__(self,
                    x_center, half_width,
                    y_center, half_height,
                    y_bar_start, half_bar_width, half_bar_height,
                    tracker_type='BOOSTING'):
        """
        Parameters
        ----------
        x_center : int
            Horizontal position of the center of the main area.
        half_width : int
            Half the width of the main area.
        y_center : int
            Vertical position of the center of the main area.
        half_height : int
            Half the height of the main area.
        y_bar_start : int
            Vertical position of the inner area. This value is relative to the
            main area.
        half_bar_width : int
            Half the width of the inner area.
        half_bar_height : int
            Half the height of the inner area.
        tracker_type : str
            Name of one of the built-in trackers in OpenCV.
        """
        SectionProcessor.__init__(self)
        # Top-left corner of the main area.
        self.x_start = int(x_center - half_width)
        self.y_start = int(y_center - half_height)

        # Bottom-right corner of the main area.
        self.x_end = int(x_center + half_width)
        self.y_end = int(y_center + half_height)

        # Top-left corner of the inner area. This position is relative to the
        # main area.
        self.y_bar_start = y_bar_start
        self.x_bar_start = int(half_width - half_bar_width)

        # Inner area
        self.bbox = (self.x_bar_start, self.y_bar_start,
                half_bar_width * 2, half_bar_height * 2)

        # Vertical positions of the top-left corner of the inner area
        self.y_pos_history = None

        # Indices of the frames where the inner area is in the bottom position.
        self.peaks = None

         # Tracker
        (major_ver, minor_ver, subminor_ver) = (cv.__version__).split('.')
        if int(minor_ver) < 3:
            self.tracker = cv.Tracker_create(tracker_type)
        else:
            if tracker_type == 'BOOSTING':
                self.tracker = cv.TrackerBoosting_create()
            if tracker_type == 'MIL':
                self.tracker = cv.TrackerMIL_create()
            if tracker_type == 'KCF':
                self.tracker = cv.TrackerKCF_create()
            if tracker_type == 'TLD':
                self.tracker = cv.TrackerTLD_create()
            if tracker_type == 'MEDIANFLOW':
                self.tracker = cv.TrackerMedianFlow_create()
            if tracker_type == 'GOTURN':
                self.tracker = cv.TrackerGOTURN_create()
            if tracker_type == 'MOSSE':
                self.tracker = cv.TrackerMOSSE_create()
            if tracker_type == "CSRT":
                self.tracker = cv.TrackerCSRT_create()

    def init(self, frame):
        """
        Initializes the tracker. The object to be tracked is defined by the
        content of the inner area in the frame

        Parameters
        ----------
        frame : ndarray
            3-channel image.
        """

        # Extract the main area.
        main_area = extract_area(frame, self.x_start, self.y_start,
            self.x_end - self.x_start, self.y_end - self.y_start)

        # Initialize the tracker with the main and inner areas.
        self.tracker.init(main_area, self.bbox)

        # Init y value of top-left corner of inner area.
        self.y_pos_history = np.array(int(self.bbox[1]))

    def process_frame(self, frame):
        """
        Processes the frame to update the inner area, that is, finding the
        position of the object defined in the initial frame.

        Parameters
        ----------
        frame : ndarray
            3-channel image.
        """
        if self.y_pos_history is None:
            raise ValueError('Tracker is not initialized')

        main_area = extract_area(frame, self.x_start, self.y_start,
            self.x_end - self.x_start, self.y_end - self.y_start)

        ok, bbox = self.tracker.update(main_area)

        if ok:
            self.bbox = bbox
            # Vertical position of the tracking object in the current frame.
            self.y_pos_history = np.hstack((self.y_pos_history, int(self.bbox[1])))

    def calculate_positions(self):
        """
        Calculates the frames where the inner area is in the bottom position.
        The result is set to the peaks attribute.
        """
        self.__calculate_press_down_positions()

    def __calculate_press_down_positions(self):
        """
        Calculates the frames where the inner area is in the bottom position.
        The result is set to the peaks attribute.
        Method just for easy reading.
        """
        if self.y_pos_history is None:
            raise ValueError('Tracker is not initialized')

        # Normalize the vertical positions of the top-left corner of the
        # bounding box of the tracking object. The limits are the initial y
        # position and 25, which is a value determined by observation
        self.y_pos_history = normalize(self.y_pos_history, self.y_bar_start, 25)

        # Find the peaks of the sine-shape curve.
        self.peaks = find_peaks(self.y_pos_history, 0.5)

    def generate_report(self, fps, sub_name):
        timestamps = np.copy(self.peaks) / fps
        func = np.vectorize(readable_time)
        np.savetxt('press_{}.csv'.format(sub_name), func(timestamps), delimiter=',', fmt='%s')

    def draw_inner_area(self, frame):
        """
        Draws the inner area (bounding box of the tracking object).

        Parameters
        ----------
        frame : ndarray
            3-channel image where the inner area will be drawn.
        """
        main_area = extract_area(frame, self.x_start, self.y_start,
            self.x_end - self.x_start, self.y_end - self.y_start)
        x, y = int(self.bbox[0]), int(self.bbox[1])
        w, h = int(self.bbox[2]), int(self.bbox[3])

        H = main_area.shape[0]
        W = main_area.shape[1]
        p1 = (x, y)
        p2 = (x + w, y + h)
        cv.rectangle(main_area, p1, p2, (255,0,0), 1, 1)

        # Resize for better visualization
        main_area = cv.resize(main_area, (W * 3, H * 3))

        cv.imshow("Tracking", main_area)
