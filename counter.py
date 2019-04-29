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

    Methods
    -------
    analyse(press_counter)
        Analyses the video to count the number of press moves.
    draw_press_counter(self, outname, frame_numbers)
        Creates a video with text of the press moves.
    """

    def __init__(self, filename, processors):
        """
        Parameters
        ----------
        filename : str
            Name of the video file to be processed.
        processors : list of SectionProcessor
            Objects that process different section of frames.
        """
        self.filename = filename
        self.processors = processors

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

        # Set init frame
        ok, frame = cap.read()
        self.__process_frame(frame, analysis, True)

        while(1):
            ok, frame = cap.read()

            if not ok:
                break

            self.__process_frame(frame, analysis)

            # Exit if ESC pressed
            k = cv.waitKey(1) & 0xff
            if k == 27:
                break

        for idx, processor in enumerate(self.processors):
            processor.calculate_positions()
            processor.plot("{}_{}_{}".format(type(processor).__name__ ,idx, time_and_date()))

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
            processor.generate_report(self.fps, '{}_{}{}'.format(idx, time_date, simple_name))

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

        frame_counter = 0
        font = cv.FONT_HERSHEY_SIMPLEX

        while(1):
            ok, frame = cap.read()

            if not ok:
                break

            # Draw the captions
            for processor in self.processors:
                processor.draw_processing_info(frame_counter, frame, font)

            out.write(frame)
            frame_counter = frame_counter + 1

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
