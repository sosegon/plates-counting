import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from utils import extract_area, readable_time
from section_processor import SectionProcessor

class LineAlarm(SectionProcessor):
    """
    A class to find the points where the plates in bands reach marks

    The points are found by thresholding the lightness in bands, finding
    the limit of the threshold in every frame, and comparing those values
    against the positions of the marks.

    To achieve the previous process, every frame is transformed using
    homographies. This transformation allows to remove the perspective and
    displaye the bands from a top-view.

    ...

    Attributes
    ----------
    src_points : ndarray
        Positions of the marks in unwarped frames.
    dst_points : ndarray
        Positions of the marks in warped frames.
    upper_left_rectangle : ((int, int), (int, int))
        Points that define the first section of the left band, from start to
        first mark.
    upper_right_rectangle : ((int, int), (int, int))
        Points that define the first section of the right band, from start to
        first mark.
    lower_left_rectangle : ((int, int), (int, int))
        Points that define the second section of the left band, from first mark
        to second mark.
    lower_right_rectangle : ((int, int), (int, int))
        Points that define the second section of the right band, from first mark
        to second mark.
    h : ndarray
        Transformation matrix to warp images and get rid of perspective.
    history_left : ndarray
        Limit of the threshold in the left in every frame.
    history_right : ndarray
        Limit of the threshold in the right every frame.
    offset : int
        Value to add to marks' positions. This allows to find the positions
        where the alarms has to be triggered.
    alarms : array
        Positions where the alarms for every mark in every band have to be
        triggered.

    Methods
    -------
    init(src_points, dst_dims)
        Calculates the transformation matrix, and start the history arrays.
    process_frame(frame)
        Processes a frame to add the threshold limit positions to the histories.
    warp(frame)
        Warp a frame using the transformation matrix.
    draw_rects(warpFrame)
        Draws the sections defined by the marks in a warped frame.
    calculate_alarms()
        Calculates the frames where the alarms have to be triggered.
    """

    def __init__(self, src_points, dst_dims):
        """
        Parameters
        ----------
        src_points : ndarray
            Positions of the marks in unwarped frames.
        dst_dims : ndarray
            Starting point and dimensions of the destination rectangle that
            would be used to do the homography.
        """
        x, y, w, h = dst_dims
        q0 = [x    , y]
        q1 = [x + w, y]
        q2 = [x    , y + h]
        q3 = [x + w, y + h]

        # Points to do the homography
        self.src_points = src_points
        self.dst_points = np.array([q0, q1, q2, q3])

        # Rectangles that define the limit to trigger the alarms
        self.upper_left_rect =  ((int(x - 2 * w + w / 2), 0), (int(x - w / 2)            , y))
        self.upper_right_rect = ((int(x + w + w / 2)    , 0), (int(x + w + 2 * w - w / 2), y))
        self.lower_left_rect =  ((int(x - 2 * w + w / 2), y), (int(x - w / 2)            , y + h))
        self.lower_right_rect = ((int(x + w + w / 2)    , y), (int(x + w + 2*w - w/2)    , y + h))

        # Transformation matrix of the homography
        self.h = None

        # Arrays to keep an history of the movement of the plates along the
        # bands
        self.history_left = np.array([])
        self.history_right = np.array([])

        # Factor to avoid None types due to the limit of the image
        self.offset = 3

        # Points (frame indices) where the alarms are triggered
        self.alarms = np.array([])

        # Points (frame indices) where marks change state
        self.events = []

    def init(self, frame):
        """
        Calculates the transformation matrix, and start the history arrays.

        Parameters
        ----------
        frame : ndarray
            3-channel image.
        """

        # Calculate the transformation Matrix
        self.h, status = cv.findHomography(self.src_points, self.dst_points)

        # Processed image and movement points
        i, l, r = self.__process_frame(frame)
        self.history_left = np.hstack((self.history_left, l))
        self.history_right = np.hstack((self.history_right, r))

    def process_frame(self, frame):
        """
        Processes a frame to add the threshold limit positions to the histories.

        Parameters
        ----------
        frame : ndarray
            3-channel image.

        Returns
        -------
        ndarray : 1-channel warped image.
        """
        if self.h is None:
            raise ValueError('Homography matrix is not initialized')

        # Processed image and movement points
        i, l, r = self.__process_frame(frame)

        # Append the new points to history
        self.history_left = np.hstack((self.history_left, l))
        self.history_right = np.hstack((self.history_right, r))

        return i


    def calculate_positions(self):
        """
        Calculates the frames where the alarms have to be triggered.
        """
        self.__remove_none()
        self.__calculate_alarms()

    def __remove_none(self):
        """
        Removes None items in histories. None items were added when processing
        each frame.
        """
        left_none = np.argwhere(self.history_left == None).ravel()
        if left_none.shape[0] > 0:
            last = left_none[-1]
            first = self.history_left[last + 1]
            self.history_left[left_none] = first

        right_none = np.argwhere(self.history_right == None).ravel()
        if right_none.shape[0] > 0:
            last = right_none[-1]
            first = self.history_right[last + 1]
            self.history_right[right_none] = first

    def __calculate_alarms(self):
        """
        Calculates the frames where the alarms have to be triggered.
        Method just for easy reading.
        """
        upper_left = np.argwhere(self.history_left >= self.upper_left_rect[1][1] - self.offset).ravel()
        upper_right = np.argwhere(self.history_right >= self.upper_right_rect[1][1] - self.offset).ravel()
        lower_left = np.argwhere(self.history_left >= self.lower_left_rect[1][1] - self.offset).ravel()
        lower_right = np.argwhere(self.history_right >= self.lower_right_rect[1][1] - self.offset).ravel()

        upper_left = self.__def_limit(upper_left)
        upper_right = self.__def_limit(upper_right)
        lower_left = self.__def_limit(lower_left)
        lower_right = self.__def_limit(lower_right)

        self.alarms = np.array([upper_left, upper_right, lower_left, lower_right])

    def generate_report(self, fps, sub_name):
        """
        Generates a report with the timestaps of the triggered alarms.
        The report is written to a file named alarms_$sub_name$.csv

        Parameters
        ----------
        fps : float
            Number of frames per second of the processed video.
        sub_name : str
            Partial name of the file report.
        """
        out_alarms = np.copy(self.alarms)
        out_alarms[out_alarms == None] = -1

        timestamps = out_alarms / fps
        func = np.vectorize(readable_time)
        records = func(timestamps)

        for idx, t in enumerate(timestamps):
            if t < 0:
                records[idx] = 'No time'

        records = np.vstack((records, ['upper_left', 'upper_right', 'lower_left', 'lower_right']))
        np.savetxt('alarms_{}.csv'.format(sub_name), records.T, delimiter=',', fmt='%s')

    def draw_caption(self, value, frame, font, position=(0, 0), color=(0, 0, 255)):
        """
        Draws the alarms in a frame.

        Parameters
        ----------
        value : list of bool
            The status of the alarms
        frame : ndarray
            3-channel image.
        font : int
            Font type available in OpenCV.
        position : (int, int)
            Position in the frame to start drawing the caption.
        color : (int, int, int)
            BGR color of the caption.
        """
        if type(value) is list:
            for idx, val in enumerate(value):
                text = '{}'.format(val)
                pos = (self.src_points[idx][0], self.src_points[idx][1])
                if val:
                    color = (0, 0, 255)
                else:
                    color = (0, 255, 0)
                super().draw_text(text, frame, font, color, pos)

    def draw_processing_info(self, frame_number, frame, font, position=(0, 0), color=(0, 0, 255)):
        """
        Draws the state of the alarms in a frame.

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
        marks_state = []
        for ev in self.events:
            state = ev[ev[:, 0] <= frame_number]
            if state.shape[0] > 0:
                marks_state.append(False if state[-1, 1] == 0 else True)
            else:
                marks_state.append(False) #unreachable statement

        self.draw_caption(marks_state, frame, font, position, color)

    def warp(self, frame):
        """
        Warp a frame using the transformation matrix.

        Parameters
        ----------
        frame : ndarray
            3-channel image.

        Returns
        -------
        ndarray : 3-channel warped image.
        """
        return cv.warpPerspective(frame, self.h, (frame.shape[1], frame.shape[0]))

    def draw_rects(self, warpFrame):
        """
        Draws the sections defined by the marks in a warped frame.

        Parameters
        ----------
        warpFrame : ndarray
            3-channel warped image.
        """
        cv.rectangle(warpFrame, self.upper_left_rect[0], self.upper_left_rect[1], (255, 0, 0), 1, 1)
        cv.rectangle(warpFrame, self.upper_right_rect[0], self.upper_right_rect[1], (255, 0, 0), 1, 1)
        cv.rectangle(warpFrame, self.lower_left_rect[0], self.lower_left_rect[1], (0, 0, 255), 1, 1)
        cv.rectangle(warpFrame, self.lower_right_rect[0], self.lower_right_rect[1], (0, 0, 255), 1, 1)

        return warpFrame

    def __process_frame(self, frame):
        """
        Process a frame to apply thresholding and add the limits to the history
        elements. A limit is valid as long as the white section reaches de end
        of the band.

        Parameters
        ----------
        frame : ndarray
            3-channel image.

        Returns
        -------
        (ndarray, int, int) : Thresheld warped image for visualization, limit of
        the left threshold, limit of the right threshold.
        """

        # Frame in zenital view
        warp = self.warp(frame)

        # Limits to threshold the image in the Light channel
        o_l, o_h = 0, 20

        # Limits to threshold the image in RGB channels to detect orange color
        # r_l, r_h, g_l, g_h, b_l, b_h = 230, 255, 115, 170, 50, 85

        # Arrays for thresheld images
        th = []
        gr = []

        # Merge the upper and lower rectangles
        box_left = (self.upper_left_rect[0], self.lower_left_rect[1])
        box_right = (self.upper_right_rect[0], self.lower_right_rect[1])

        # In every rectangle...
        for idx, box in enumerate([box_left, box_right]):
            # Box of interest
            boi = extract_area(warp,
                box[0][0],
                box[0][1],
                box[1][0] - box[0][0],
                box[1][1] - box[0][1])
            # HLS
            boi = cv.cvtColor(boi, cv.COLOR_BGR2HLS)
            # Lightness channel
            boi = boi[:,:,1]
            # Pixels within the threshold limits
            bools = (boi > o_l) & (boi < o_h)
            # Thresheld box
            binary = np.zeros_like(boi)

            # b = boi[:,:,0]
            # g = boi[:,:,1]
            # r = boi[:,:,2]
            # bools = (r > r_l) & (r < r_h) & (g > g_l) & (g < g_h) & (b > b_l) & (b < b_h)
            # binary = np.zeros_like(boi[:,:,0])

            # Valid pixels as white
            binary[bools == True] = 1

            # Find the  white pixels
            whites = np.argwhere(binary == 1)

            left_nones = []
            right_nones = []

            if whites.shape[0] > 0:
                # fill all space between first and last white pixel
                rows = whites[:, 0]
                rows = np.unique(rows)
                low = rows[0]
                high = rows[-1] + 1
                binary[low:high, :] = 1

                # Find the first white pixel
                # If the box of whites extends to the end of the band, add the
                # first white pixel
                if boi.shape[0] - high < self.offset:
                    if idx == 0:
                        max_left = low

                    elif idx == 1:
                        max_right = low
                # If the box of whites does not extend to the end of the band
                else:
                    if idx == 0 and self.history_left.shape[0] > 0:
                        max_left = self.history_left[-1]

                    elif idx == 0 and self.history_left.shape[0] == 0:
                        max_left = None

                    if idx == 1 and self.history_right.shape[0] > 0:
                        max_right = self.history_right[-1]

                    elif idx == 1 and self.history_right.shape[0] == 0:
                        max_right = None
            else:
                if idx == 0 and self.history_left.shape[0] > 0:
                    max_left = self.history_left[-1]

                elif idx == 0 and self.history_left.shape[0] == 0:
                    max_left = None

                if idx == 1 and self.history_right.shape[0] > 0:
                    max_right = self.history_right[-1]

                elif idx == 1 and self.history_right.shape[0] == 0:
                    max_right = None

            th.append(binary)

            # Grayscale image
            box = extract_area(warp,
                box[0][0],
                box[0][1],
                box[1][0] - box[0][0],
                box[1][1] - box[0][1])
            box = cv.cvtColor(box, cv.COLOR_BGR2RGB)
            gr.append(box)

        # Left and right thresheld bands
        th_image = np.hstack((th[0], th[1]))*255
        th_image = cv.resize(th_image,  (th_image.shape[1]*3, th_image.shape[0]*3))

        # left and right grayscale bands
        gr_image = np.hstack((gr[0], gr[1]))*255

        # Limit lines drawn in the grayscale bands
        gr_image = cv.cvtColor(gr_image, cv.COLOR_BGR2GRAY)
        upper = self.upper_left_rect[1][1] - self.offset
        lower = self.lower_left_rect[1][1] - self.offset
        gr_image[upper,:] = 0
        gr_image[lower,:] = 0
        gr_image = cv.resize(gr_image,  (gr_image.shape[1]*3, gr_image.shape[0]*3))

        # Merge thresheld and grayscale images
        gr_image = np.hstack((th_image, gr_image))

        return gr_image.T, max_left, max_right

    def __def_limit(self, indices):
        """
        Define the index of the position where an alarm has to be triggered.

        Parameters
        ----------
        indices : ndarray
            Indices of the frames where the mark was passed by the threshold

        Returns
        -------
            int : Unique index of first frame where plates reached the mark.
        """
        if indices.shape[0] > 0:
            return indices[0]
        else:
            return None

    def show_processing(self, frame, name="Alarms"):
        """
        Displays the thresholding processing of the bands.

        Parameters
        ----------
        frame : ndarray
            3-channel image.
        name : str
            Name of the window to display the frame.
        """
        cv.imshow(name, self.__process_frame(frame)[0])

    def plot(self, name="Alarms"):
        """
        Plots the position of the plates in the band. The 'X's correspond to
        the positions of the alarms.

        Parameters
        ----------
        name : str
            Name of the plot.
        """
        plt.plot(self.history_left)
        plt.plot(self.history_right)

        for idx, ev in enumerate(self.events):
            if idx % 2 == 0:
                plt.plot(ev[:, 0], self.history_left[ev[:, 0]], 'X')
            else:
                plt.plot(ev[:, 0], self.history_right[ev[:, 0]], 'X')
        plt.savefig(name)
        plt.figure()

    def calculate_events(self, fps, last_frame):
        """
        Calculate the points (frame indices) where the marks change their
        states. A mark is either True or False. A mark is False as long as it is
        not reached by the line of plates, otherwise is True.

        Parameters
        ----------
        fps : float
            Frames per second of the processed video.
        last_frame : int
            Index of the last frame in the processed video
        """
        limits = [self.upper_left_rect[1][1] - self.offset,
        self.upper_right_rect[1][1] - self.offset,
        self.lower_left_rect[1][1] - self.offset,
        self.lower_right_rect[1][1] - self.offset]

        for idx, limit in enumerate(limits):
            if idx % 2 == 0:
                state = self.__state_function(self.history_left, limit, fps)
                self.events.append(self.__changes(state, last_frame))
            else:
                state = self.__state_function(self.history_right, limit, fps)
                self.events.append(self.__changes(state, last_frame))

    def __state_function(self, sequence, limit, fps):
        """
        Creates a signal with values either 0 or 1. The sections try to be as
        long as possible to avoid quick changes in state.

        Parameters
        ----------
        sequence : ndarray
            List of raw points for the function
        limit : int
            Value which define the condition to create the 0 and 1 values.
        fps : float
            Frames per second of the processed video.

        Returns
        -------
        ndarray : Signal with values 0 or 1.
        """
        bools = sequence > limit
        bools = bools.astype(int)
        plt.plot(bools)

        ones_index = np.argwhere(bools == 1).ravel()

        diff = np.diff(ones_index)

        for idx, d in enumerate(diff):
            if d > 1 and d < 2 * fps:
                one_index = ones_index[idx]
                bools[one_index:one_index + d] = 1

        return bools

    def __changes(self, state_list, last_frame):
        """
        Calculates and sets changes in the state of a signal.

        Parameters
        ----------
        state_list : ndarray
            Signal of states.
        last_frame : int
            Index of the last frame of the processed video

        Returns
        -------
        ndarray : Indices of the frames where the state changes, and the values
        of change.
        """
        diff =  np.diff(state_list)

        passed = np.argwhere(diff > 0).ravel()
        passed_state = np.ones(passed.shape[0])

        unpassed = np.argwhere(diff < 0).ravel()
        unpassed_state = np.zeros(unpassed.shape[0])

        indices = np.hstack((passed, unpassed))
        states = np.hstack((passed_state, unpassed_state))
        total = np.vstack((indices, states)).T

        if total.shape[0] > 0:
            total = total[total[:, 0].argsort()]

            first = 1 if total[0][1] == 0 else 0
            last = total[-1][1]

            total = np.vstack(([0, first], total, [last_frame, last]))
        else:
            total = np.array([[0, state_list[0]],[last_frame, state_list[0]]])

        return total.astype(int)
