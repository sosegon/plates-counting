import cv2 as cv
import matplotlib.pyplot as plt
from press_counter import PressCounter
from plates_counter import PlatesCounter
from line_alarm import LineAlarm
from utils import get_file_simple_name, time_and_date

class Counter:
    """
    A class to count press moves from a video of a manufacturing
    machine.

    ...

    Attributes
    ----------
    filename : str
        Name of the video file to be processed.

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
        """
        self.filename = filename
        self.processors = processors

    def analyse(self, analysis=False):
        """
        Analyses the video to count the number of press moves.

        Parameters
        ----------
        processors : list of SectionProcessor
            Objects that process different section of frames.
        analysis : bool
            Flag used for analysis during development.
        """

        cap = cv.VideoCapture(self.filename)
        self.__calculate_fps(cap)

        # Set init frame
        ok, frame = cap.read()
        for processor in self.processors:
            processor.init(frame)

            if analysis and type(processor).__name__ == 'PressCounter':
                processor.draw_inner_area(frame)
                sleep(0.0)

        while(1):
            ok, frame = cap.read()

            if not ok:
                break

            for processor in self.processors:
                processor.process_frame(frame)

                if analysis and type(processor).__name__ == 'PressCounter':
                    # Slow down the movement for better visualization.
                    sleep(0.0)
                    processor.draw_inner_area(frame)

            # Exit if ESC pressed
            k = cv.waitKey(1) & 0xff
            if k == 27:
                break

        for processor in self.processors:
            processor.calculate_positions()

        if analysis:
            for idx, processor in enumerate(self.processors):
                if type(processor).__name__ == 'PressCounter':
                    peaks = processor.peaks
                    plt.plot(processor.y_pos_history)
                    plt.plot(peaks, processor.y_pos_history[peaks], 'X');
                    plt.savefig('press')
                    plt.figure()

                if type(processor).__name__ == 'PlatesCounter':
                    peaks = processor.peaks
                    plt.plot(processor.light_history)
                    plt.plot(peaks, processor.light_history[peaks], 'X')
                    plt.savefig('plates{:d}'.format(idx))
                    plt.figure()

            cv.destroyAllWindows()

        cap.release()

    def generate_report(self):
        """
        Generates report for every SectionProcessor.

        Parameters
        ----------
        processors : list of SectionProcessor.
            Objects that processed different section of frames.
        """
        simple_name = get_file_simple_name(self.filename)
        time_date = time_and_date()

        for idx, processor in enumerate(self.processors):
            processor.generate_report(self.fps, '{}_{}{}'.format(idx, time_date, simple_name))

    def draw_press_counter(self, outname):
        """
        Creates a video with text of the press moves.

        Parameters
        ----------
        outname : str
            Name of the video file to be created.
        frames_press : ndarray
            Indices of the frames where the press is in the bottom position.
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

        # Frame indices for captions
        frames_press = [p.peaks for p in self.processors if type(p) is PressCounter][0]
        frames_plate_list = [p.peaks for p in self.processors if type(p) is PlatesCounter]
        alarms = [p.alarms for p in self.processors if type(p) is LineAlarm][0]

        # Counters to define the positions in the video for press moves.
        frame_counter = 0
        press_moves = 0

        # Counter for every PlatesCounter processor
        plate_counts = []
        for frames_plate in frames_plate_list:
            plate_counts.append(0)

        # Initially, no alarm has been triggered
        alarms_reached = [False, False, False, False]

        font = cv.FONT_HERSHEY_SIMPLEX

        while(1):
            ok, frame = cap.read()

            if not ok:
                break

            # Calculate the number of press moves for the current frame
            if len(frames_press) > 0 and frame_counter == frames_press[0]:
                press_moves = press_moves + 1
                frames_press = frames_press[1:]

            # Calculate the number of plates for the current frame
            for idx, frames_plate in enumerate(frames_plate_list):
                if len(frames_plate) > 0 and frame_counter == frames_plate[0]:
                    plate_counts[idx] = plate_counts[idx] + 1
                    frames_plate_list[idx] = frames_plate[1:]

            # Set the alarms for the current frame
            for idx, flag in enumerate(alarms_reached):
                if not flag and alarms[idx] == frame_counter:
                    alarms_reached[idx] = True

            # Identify the PlatesCounter objects
            platesCounter_counter = 0
            # Draw the captions
            for processor in self.processors:
                processor_type = type(processor)

                if processor_type is PressCounter:
                    processor.draw_caption(press_moves, frame, font)

                elif processor_type is PlatesCounter:
                    processor.draw_caption(plate_counts[platesCounter_counter], frame, font)
                    platesCounter_counter = platesCounter_counter + 1

                elif processor_type is LineAlarm:
                    processor.draw_caption(alarms_reached, frame, font)

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
