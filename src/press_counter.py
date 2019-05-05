import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
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
    events : ndarray
        Positions where the state changes.

    Methods
    -------
    init_tracker(frame)
        Initializes the tracker.
    process_frame(frame)
        Processes the frame to update the inner area.
    calculate_positions()
        Calculates the frames where the inner area is in the bottom position.
    generate_report(fps, sub_name):
        Generates a report with the timestaps of the press moves.
    draw_caption(value, frame, font, position=(0, 0), color=(0, 0, 255))
        Draws the number of press moves in a frame.
    draw_processing_info(frame_number, frame, font, position=(0, 0), color=(0, 0, 255))
        Draws the number of press moves in a frame.
    show_processing(frame, name="Press")
        Displays the inner area (bounding box of the tracking object).
    plot(name="Press")
        Plots the positions of the press in every frame.
    calculate_events(fps, last_frame)
        Calculate the points where the press changes its state.
    draw_inner_area(frame)
        Draws the inner area.
    state_at_frame(frame_number)
        Returns the state of the processor at a given frame.
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

        self.events = None

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
        """
        Generates a report with the timestaps of the press moves.
        The report is written to a file named press_$sub_name$.csv

        Parameters
        ----------
        fps : float
            Number of frames per second of the processed video.
        sub_name : str
            Partial name of the file report.
        """
        timestamps = np.copy(self.peaks) / fps
        file_name = 'press_{}.csv'.format(sub_name)

        if timestamps.shape[0] > 0:
            func = np.vectorize(readable_time)
            array_to_dump = func(timestamps)
        else:
            array_to_dump = np.array(['No press moves'])

        np.savetxt(file_name, array_to_dump, delimiter=',', fmt='%s')


    def draw_caption(self, value, frame, font, position=(0, 0), color=(0, 0, 255)):
        """
        Draws the number of press moves in a frame.

        Parameters
        ----------
        value : int
            Number of press moves.
        frame : ndarray
            3-channel image.
        font : int
            Font type available in OpenCV.
        position : (int, int)
            Position in the frame to start drawing the caption.
        color : (int, int, int)
            BGR color of the caption.
        """
        text = 'Press: {}'.format(value)
        super().draw_text(text, frame, font, color, (self.x_start, self.y_start))

    def draw_processing_info(self, frame_number, frame, font, position=(0, 0), color=(0, 0, 255)):
        """
        Draws the number of press moves in a frame.

        Parameters
        ----------
        frame_number : int
            Value to compare to internal information of processor.
        frame : ndarray
            3-channel image.
        font : int
            Font type available in OpenCV.
        position : (int, int)
            Position in the frame to start drawing the information.
        color : (int, int, int)
            BGR color of the information.
        """
        valid_frames = self.peaks[self.peaks <= frame_number]

        self.draw_caption(valid_frames.shape[0], frame, font, position, color)

    def show_processing(self, frame, name="Press"):
        """
        Displays the inner area (bounding box of the tracking object).

        Parameters
        ----------
        frame : ndarray
            3-channel image.
        name : str
            Name of the window to display the frame.
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

        cv.imshow(name, main_area)

    def plot(self, name="Press"):
        """
        Plots the positions of the press in every frame. The blue 'X's
        correspond to the press in the bottom position. Green 'X's correspond
        to the press in ON state. Red 'X's correspond to the press in OFF state.

        Parameters
        ----------
        name : str
            Name of the plot.
        """
        plt.plot(self.y_pos_history)
        plt.plot(self.peaks, self.y_pos_history[self.peaks], 'X', color='blue')

        true_events = self.events[self.events[:, 1] == 1]
        false_events = self.events[self.events[:, 1] == 0]
        plt.plot(true_events[:, 0], self.y_pos_history[true_events[:, 0]], 'X', color='green')
        plt.plot(false_events[:, 0], self.y_pos_history[false_events[:, 0]], 'X', color='red')

        plt.savefig(name)
        plt.figure()

    def calculate_events(self, fps, last_frame):
        """
        Calculate the points (frame indices) where the press changes its state.
        The press is either ON or OFF. The press is ON as long as it is moving,
        which is determined by analysing the peaks (down positions). Basically,
        if the distance between peaks is less than 2 seconds, the press is
        moving.

        Parameters
        ----------
        fps : float
            Frames per second of the processed video.
        last_frame : int
            Index of the last frame in the processed video
        """

        if self.peaks is None:
            raise ValueError('Peaks not calculated')

        self.events = np.array([[-1, -1]])

        # No peaks, no processing
        if self.peaks.shape[0] == 0:
            return

        threshold_seconds = 3
        prev_frame = 0

        # Define the first event at frame 0
        diff = self.peaks[0] - prev_frame

        # The first event is either ON or OFF. It is ON if the time between
        # frame 0 and the first peak is least than 3 seconds.
        if diff < threshold_seconds * fps:
            self.events = np.vstack((self.events, [prev_frame, 1]))
        else:
            self.events = np.vstack((self.events, [prev_frame, 0]))

        # Remove the initial event, which was just for compatibility
        self.events = self.events[1:, :]

        # Iterate only if there are more than one peak
        if self.peaks.shape[0] > 1:
            prev_frame = self.peaks[0]

            # Iterate over the rest of peaks
            for frame in self.peaks[1:]:

                diff = frame - prev_frame
                prev_event = self.events[-1, 1]

                if diff < threshold_seconds * fps:
                    if prev_event == 1:
                        prev_frame = frame
                        continue
                    else:
                        self.events = np.vstack((self.events, [prev_frame, 1]))
                else:
                    if prev_event == 0:
                        prev_frame = frame
                        continue
                    else:
                        self.events = np.vstack((self.events, [prev_frame, 0]))

                prev_frame = frame


            # Check the last peak
            diff = last_frame - self.peaks[-1]
            prev_event = self.events[-1, 1]

            if diff < threshold_seconds * fps:
                if prev_event == 1:
                    pass
                else:
                    self.events = np.vstack((self.events, [self.peaks[-1], 1]))
            else:
                if prev_event == 0:
                    pass
                else:
                    self.events = np.vstack((self.events, [self.peaks[-1], 0]))

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
        ndarray : The state at a given frame: ON/OFF + press moves
        """
        press_indices = np.argwhere(self.peaks <= frame_number).ravel()
        state = press_indices.shape[0]

        event_indices = np.argwhere(self.events[:, 0] <= frame_number).ravel()

        if event_indices.shape[0] > 0:
            change = self.events[event_indices[-1], 1]
        else:
            change = -1

        state = np.hstack((change, state))

        return state
