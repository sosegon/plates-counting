import argparse
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from time import sleep, time
from scipy.signal import find_peaks

class Counter:
    """
    A class to count press moves and plates from a video of a manufacturing
    machine.

    ...

    Attributes
    ----------
    filename : str
        Name of the video file to be processed.

    Methods
    -------
    analyse(x_center, half_width, y_center, half_height, y_bar_start,
    half_bar_width, half_bar_height, tracker_type='BOOSTING')
        Performs the analysis to count the press moves.
    """

    def __init__(self, filename, outname="out.mp4"):
        """
        Parameters
        ----------
        filename : str
            Name of the video file to be processed.
        outname : str
            Name of the output video.
        """
        self.filename = filename
        self.outname = outname

    def analyse(self,
                    x_center, half_width,
                    y_center, half_height,
                    y_bar_start, half_bar_width, half_bar_height,
                    tracker_type='BOOSTING', analysis=False):
        """
        Processes the video to count the number of press moves.

        The processing is done by tracking the press. To do so, it is required
        to define two areas. The main area corresponds to a section of the
        machine where the tracking will be performed. The second area is
        contained within the main one. This area corresponds to the object that
        will be tracked.

        The tracking is performed with a built-in object tracker in OpenCV.

        During the tracking, the position of the top-left corner of the bounding
        box of the press is recored in every frame. This process creates a set
        of points. Those points form a sine-shape curve. The peaks of that curve
        correspond to the lower position of the press. The number of peaks
        corresponds to the number of press moves.

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
            Vertical position of the inner area. This value is relative to
            the main area.
        half_bar_width : int
            Half the width of the inner area.
        half_bar_height : int
            Half the height of the inner area.
        tracker_type : str
            Name of one of the built-in trackers in OpenCV.
        analysis : bool
            Flag used for analysis during development.
        """

        cap = cv.VideoCapture(self.filename)

        # Top-left corner of the main area.
        x_start = int(x_center - half_width)
        y_start = int(y_center - half_height)

        # Bottom-right corner of the main area.
        x_end = int(x_center + half_width)
        y_end = int(y_center + half_height)

        # Dimensions of the main area.
        width = int(half_width * 2)
        height = int(half_height * 2)

        # Top-left corner of the inner area, along with y_bar_start. This
        # position is relative to the main area.
        x_bar_start = int(half_width - half_bar_width)

        # Bottom-right corner of the inner area. This position is relative to
        # the main area.
        x_bar_end = int(x_bar_start + half_bar_width * 2)
        y_bar_end = int(y_bar_start + half_bar_height * 2)

        # Inner area
        bbox = (x_bar_start, y_bar_start, half_bar_width * 2, half_bar_height * 2)

        # Tracker
        (major_ver, minor_ver, subminor_ver) = (cv.__version__).split('.')
        if int(minor_ver) < 3:
            tracker = cv.Tracker_create(tracker_type)
        else:
            if tracker_type == 'BOOSTING':
                tracker = cv.TrackerBoosting_create()
            if tracker_type == 'MIL':
                tracker = cv.TrackerMIL_create()
            if tracker_type == 'KCF':
                tracker = cv.TrackerKCF_create()
            if tracker_type == 'TLD':
                tracker = cv.TrackerTLD_create()
            if tracker_type == 'MEDIANFLOW':
                tracker = cv.TrackerMedianFlow_create()
            if tracker_type == 'GOTURN':
                tracker = cv.TrackerGOTURN_create()
            if tracker_type == 'MOSSE':
                tracker = cv.TrackerMOSSE_create()
            if tracker_type == "CSRT":
                tracker = cv.TrackerCSRT_create()

        # Read first frame and extact the main area.
        ok, frame = cap.read()
        frame = frame[y_start:y_end, x_start:x_end, :]

        # Initialize the tracker with the main and inner areas.
        ok = tracker.init(frame, bbox)

        # Initial vertical position of the inner area.
        y_pos_history = np.array(int(bbox[1]))

        if analysis:
            # Draw bounding box of the tracking object
            self.__draw_inner_area(
                frame,
                int(bbox[0]), int(bbox[1]),
                int(bbox[2]), int(bbox[3])
            )
            sleep(0)

        while(1):

            if analysis:
                # Slow down the movement for better visualization.
                sleep(0.0)

            ok, frame = cap.read()

            if not ok:
                break

            # Extract the main area.
            frame = frame[y_start:y_end, x_start:x_end, :]

            # Start timer
            timer = cv.getTickCount()

            # Find the object defined by the inner area in the first frame.
            ok, bbox = tracker.update(frame)

            # Frames per second.
            fps = cv.getTickFrequency() / (cv.getTickCount() - timer);

            if ok:
                # Vertical position of the tracking object in the current frame.
                y_pos_history = np.hstack((y_pos_history, int(bbox[1])))

                if analysis:
                    # Draw bounding box of the tracking object.
                    self.__draw_inner_area(
                        frame,
                        int(bbox[0]), int(bbox[1]),
                        int(bbox[2]), int(bbox[3])
                    )

            # Exit if ESC pressed
            k = cv.waitKey(1) & 0xff
            if k == 27:
                break

        # Normalize the vertical positions of the top-left corner of the
        # bounding box of the tracking object.
        min_ = np.min(y_pos_history)
        max_ = np.max(y_pos_history)

        y_pos_history = y_pos_history - min_
        y_pos_history = y_pos_history / (max_ - min_)

        if analysis:
            plt.plot(y_pos_history)

        # Find the peaks of the sine-shape curve.
        peaks, _ = find_peaks(y_pos_history, distance=15)

        # Filter the peaks to get just those corresponding to the press in the
        # down position.
        valid_peaks = np.argwhere(y_pos_history[peaks] > 0.5)
        peaks = peaks[valid_peaks.ravel()]

        if analysis:
            plt.plot(peaks, y_pos_history[peaks], 'X')
            plt.savefig('output')

        cv.destroyAllWindows()
        cap.release()

        # Draw text to coun the press moves in the video
        self.draw_press_counter(peaks)

    def draw_press_counter(self, frame_numbers):
        """
        Creates a video with text of the press moves.

        Parameters
        ----------
        frame_numbers : ndarray
            Indices of the frames where the press is in the bottom position.
        """

        cap = cv.VideoCapture(self.filename)

        # Dimensions of the input video.
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))

        # FPS in the input video.
        (major_ver, minor_ver, subminor_ver) = (cv.__version__).split('.')
        if int(major_ver)  < 3 :
            fps = cap.get(cv.cv.CV_CAP_PROP_FPS)
            print("Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps))
        else :
            fps = cap.get(cv.CAP_PROP_FPS)
            print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))

        # Output video.
        out = cv.VideoWriter(
            self.outname,
            cv.VideoWriter_fourcc('H','2','6','4'),
            fps,
            (frame_width, frame_height))

        # Counters to define the positions in the video for press moves.
        frame_counter = 0
        press_moves = 0

        font = cv.FONT_HERSHEY_SIMPLEX

        while(1):
            ok, frame = cap.read()

            if not ok:
                break

            # Increase the number of press moves.
            if len(frame_numbers) > 0 and frame_counter == frame_numbers[0]:
                press_moves = press_moves + 1
                frame_numbers = frame_numbers[1:]

            cv.putText(
                frame,
                'Press moved: {}'.format(press_moves),
                (40, 40),
                font,
                0.5,
                (255, 255, 255),
                2,
                cv.LINE_AA)

            out.write(frame)
            frame_counter = frame_counter + 1

        out.release()
        cap.release()

    def __draw_inner_area(self, frame, x, y, w, h):
        """
        Draws the inner area (bounding box of the tracking object).

        Parameters
        ----------
        frame : ndarray
            3-channel image where the inner area will be drawn.
        x : int
            Horizontal position of the top-left corner.
        y : int
            Vertical position of the top-left corner.
        w : int
            Width of the bounding box.
        h : int
            Height of the bounding box.

        """
        H = frame.shape[0]
        W = frame.shape[1]
        p1 = (x, y)
        p2 = (x + w, y + h)
        cv.rectangle(frame, p1, p2, (255,0,0), 1, 1)

        # Resize for better visualization
        frame = cv.resize(frame, (W * 3, H * 3))

        cv.imshow("Tracking", frame)

def main():
    parser = argparse.ArgumentParser(description='plates')
    parser.add_argument('filename', type=str, help='Input video filename')
    parser.add_argument('outname', type=str, help='Output video filename')

    parser.add_argument('-xc', dest='x_center', type=int, default=188,
        help='Horizontal position of the center of the main area.')

    parser.add_argument('-hw', dest='half_width', type=int, default=40,
        help='Half the width of the main area.')

    parser.add_argument('-yc', dest='y_center', type=int, default=118,
        help='Vertical position of the center of the main area.')

    parser.add_argument('-hh', dest='half_height', type=int, default=24,
        help='Half the height of the main area.')

    parser.add_argument('-yb', dest='y_bar_start', type=int, default=8,
        help='Vertical position of the inner area. This value is relative to the main area.')

    parser.add_argument('-bw', dest='half_bar_width', type=int, default=17,
        help='Half the width of the inner area.')

    parser.add_argument('-bh', dest='half_bar_height', type=int, default=12,
        help='Half the height of the inner area.')

    parser.add_argument('-tt', dest='tracker_type', type=str, default='BOOSTING',
        help='Name of one of the built-in trackers in OpenCV.')

    parser.add_argument('-a', dest='analysis', type=bool, default=False,
        help='Flag used for analysis during development.')

    args = parser.parse_args()

    filename = args.filename
    outname = args.outname

    xc = args.x_center
    hw = args.half_width
    yc = args.y_center
    hh = args.half_height
    yb = args.y_bar_start
    bw = args.half_bar_width
    bh = args.half_bar_height
    tracker_type = args.tracker_type
    analysis = args.analysis

    counter = Counter(filename, outname)

    start = time()
    counter.analyse(xc, hw, yc, hh, yb, bw, bh, tracker_type, analysis)
    end = time()

    print("Time to process: {:d}s".format(int(end - start)))

if __name__ == '__main__':
    main()