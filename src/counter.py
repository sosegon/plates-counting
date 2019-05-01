import cv2 as cv
import matplotlib.pyplot as plt
from press_counter import PressCounter
from plates_counter import PlatesCounter
from line_alarm import LineAlarm
from utils import get_file_simple_name, time_and_date
from time import sleep

class Counter:
    """
    A class to count press moves from a video of a manufacturing
    machine.

    ...

    Attributes
    ----------
    filename : str
        Name of the video file to be processed.
    processors : list of SectionProcessor
        Objects that process different section of frames.
    time_range : (float, float)
        Time in seconds for the section of the video to be processed.
    fps : float
        Frames per second of the video.
    start_frame : int
        Initial point to process the video. Calculated from time_range.
    end_frame : int
        End point to process the video. Calculated from time_range.

    Methods
    -------
    analyse(press_counter)
        Analyses the video to count the number of press moves.
    generate_report()
        Generates report for every SectionProcessor.
    draw_press_counter(self, outname, frame_numbers)
        Creates a video with text of the press moves.
    """

    def __init__(self, filename, processors, time_range):
        """
        Parameters
        ----------
        filename : str
            Name of the video file to be processed.
        processors : list of SectionProcessor
            Objects that process different section of frames.
        time_range : (int, int)
            Time in seconds for the section of the video to be processed.
        """
        self.filename = filename
        self.processors = processors
        self.time_range = time_range
        self.fps = 0
        self.start_frame = -1
        self.end_frame = -1

    def analyse(self, analysis=False):
        """
        Analyses the video to count the number of press moves.

        Parameters
        ----------
        analysis : bool
            Flag used for analysis during development.
        """

        cap = cv.VideoCapture(self.filename)
        self.__calculate_fps(cap)

        self.__calculate_start_end_frames(cap)
        frame_counter = 0

        # Skip frames until reaching the start_frame
        while(frame_counter < self.start_frame):
            ok, frame = cap.read()

            if not ok:
                break

            frame_counter = frame_counter + 1

        # Set init frame
        ok, frame = cap.read()
        self.__process_frame(frame, analysis, True)
        frame_counter = frame_counter + 1

        # Process frames until reaching the end_frame
        while(frame_counter <= self.end_frame):
            ok, frame = cap.read()

            if not ok:
                break

            self.__process_frame(frame, analysis)
            frame_counter = frame_counter + 1

            # Exit if ESC pressed
            k = cv.waitKey(1) & 0xff
            if k == 27:
                break

        for idx, processor in enumerate(self.processors):
            simple_name = get_file_simple_name(self.filename)
            processor.calculate_positions()
            processor.plot("{}_{}_{}_{}.png".format(type(processor).__name__ ,idx, time_and_date(), simple_name))

            cv.destroyAllWindows()

        cap.release()

    def __process_frame(self, frame, analysis=False, init=False):
        """
        Calls each processor to do its job.

        Parameters
        ----------
        frame : ndarray
            3-channel image.
        analysis : bool
            Flag to display the processing done in each frame.
        init : bool
            Flag to initialize each processor.
        """
        for idx, processor in enumerate(self.processors):
            if init:
                processor.init(frame)
            else:
                processor.process_frame(frame)

            if analysis:
                # Slow down the movement for better visualization.
                sleep(0.0)
                processor.show_processing(frame, "Processor {}".format(idx))

    def generate_report(self):
        """
        Generates report for every SectionProcessor.
        """
        simple_name = get_file_simple_name(self.filename)
        time_date = time_and_date()

        for idx, processor in enumerate(self.processors):
            processor.generate_report(self.fps, '{}_{}_{}'.format(idx, time_date, simple_name))

    def draw_press_counter(self, outname):
        """
        Creates a video with text for every processor.

        Parameters
        ----------
        outname : str
            Name of the video file to be created.
        """
        cap = cv.VideoCapture(self.filename)

        # Dimensions of the input video.
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))

        # Output video.
        out = cv.VideoWriter(
            outname,
            cv.VideoWriter_fourcc('H','2','6','4'),
            self.fps,
            (frame_width, frame_height))

        font = cv.FONT_HERSHEY_SIMPLEX
        frame_counter = 0

        # Skip frames until reaching the start_frame
        while(frame_counter < self.start_frame):
            ok, frame = cap.read()

            if not ok:
                break

            frame_counter = frame_counter + 1

        frame_counter_processor = 0
        # Process frames until reaching the end_frame
        while(frame_counter <= self.end_frame):
            ok, frame = cap.read()

            if not ok:
                break

            # Draw the captions
            for processor in self.processors:
                processor.draw_processing_info(frame_counter_processor, frame, font)

            out.write(frame)
            frame_counter = frame_counter + 1
            frame_counter_processor = frame_counter_processor + 1

        out.release()
        cap.release()

    def __calculate_fps(self, capture):
        """
        Calculates and sets the frames per second of a video.

        Parameters
        ----------
        capture : VideoCapture
            Video capture.
        """
        # FPS in the input video.
        (major_ver, minor_ver, subminor_ver) = (cv.__version__).split('.')
        if int(major_ver)  < 3 :
            self.fps = capture.get(cv.cv.CV_CAP_PROP_FPS)
            print("Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(self.fps))
        else :
            self.fps = capture.get(cv.CAP_PROP_FPS)
            print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(self.fps))

    def __calculate_start_end_frames(self, capture):
        """
        Calculates and sets that start and end frame of a video to be processed.

        Parameters
        ----------
        capture : VideoCapture
            Video capture.
        """
        if self.fps <= 0:
            self.__calculate_fps(capture)

        (major_ver, minor_ver, subminor_ver) = (cv.__version__).split('.')
        if int(major_ver)  < 3 :
            total_frames = capture.get(cv.cv.CV_CAP_PROP_FRAME_COUNT)
        else :
            total_frames = capture.get(cv.CAP_PROP_FRAME_COUNT)

        start_seconds, end_seconds = self.time_range

        self.start_frame = 0
        self.end_frame = total_frames - 1

        total_seconds = self.end_frame / self.fps

        if end_seconds > start_seconds and end_seconds < total_seconds:
            self.end_frame = int(end_seconds * self.fps)
            self.start_frame = max(0, int(start_seconds * self.fps))
        elif end_seconds > start_seconds and start_seconds < total_seconds:
            self.start_frame = max(0, int(start_seconds * self.fps))
