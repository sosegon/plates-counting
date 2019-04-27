import cv2 as cv
import matplotlib.pyplot as plt
from press_counter import PressCounter
from plates_counter import PlatesCounter
from line_alarm import LineAlarm

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

    def __init__(self, filename):
        """
        Parameters
        ----------
        filename : str
            Name of the video file to be processed.
        """
        self.filename = filename


    def analyse(self, press_counter, plates_counter_list, line_alarm, analysis=False):
        """
        Analyses the video to count the number of press moves.

        Parameters
        ----------
        press_counter : PressCounter
            Object in charge of count the press moves.
        analysis : bool
            Flag used for analysis during development.
        """

        cap = cv.VideoCapture(self.filename)

        # Set init frame
        ok, frame = cap.read()
        line_alarm.init(frame)
        press_counter.init_tracker(frame)
        for plates_counter in plates_counter_list:
            plates_counter.init(frame)

        if analysis:
            press_counter.draw_inner_area(frame)
            sleep(0.0)

        while(1):

            ok, frame = cap.read()

            if not ok:
                break

            line_alarm.process_frame(frame)
            press_counter.process_frame(frame)
            for plates_counter in plates_counter_list:
                plates_counter.process_frame(frame)

            if analysis:
                # Slow down the movement for better visualization.
                sleep(0.0)
                press_counter.draw_inner_area(frame)

            # Exit if ESC pressed
            k = cv.waitKey(1) & 0xff
            if k == 27:
                break

        line_alarm.calculate_alarms()
        press_counter.calculate_press_down_positions()
        for plates_counter in plates_counter_list:
            plates_counter.calculate_plates()

        if analysis:
            peaks = press_counter.peaks
            plt.plot(press_counter.y_pos_history)
            plt.plot(peaks, press_counter.y_pos_history[peaks], 'X');
            plt.savefig('press')
            plt.figure()

            for idx, plates_counter in enumerate(plates_counter_list):
                peaks = plates_counter.peaks
                plt.plot(plates_counter.light_history)
                plt.plot(peaks, plates_counter.light_history[peaks], 'X')
                plt.savefig('plates{:d}'.format(idx))
                plt.figure()

            cv.destroyAllWindows()

        cap.release()

    def draw_press_counter(self, outname, frames_press, frames_plate_list, alarms):
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
            outname,
            cv.VideoWriter_fourcc('H','2','6','4'),
            fps,
            (frame_width, frame_height))

        # Counters to define the positions in the video for press moves.
        frame_counter = 0
        press_moves = 0

        plate_counts = []
        for frames_plate in frames_plate_list:
            plate_counts.append(0)

        alarms_reached = [False, False, False, False]

        font = cv.FONT_HERSHEY_SIMPLEX

        while(1):
            ok, frame = cap.read()

            if not ok:
                break

            # Increase the number of press moves.
            if len(frames_press) > 0 and frame_counter == frames_press[0]:
                press_moves = press_moves + 1
                frames_press = frames_press[1:]

            for idx, frames_plate in enumerate(frames_plate_list):
                if len(frames_plate) > 0 and frame_counter == frames_plate[0]:
                    plate_counts[idx] = plate_counts[idx] + 1
                    frames_plate_list[idx] = frames_plate[1:]

            cv.putText(
                frame,
                'Press moved: {}'.format(press_moves),
                (40, 40),
                font,
                0.5,
                (255, 255, 255),
                2,
                cv.LINE_AA)

            for idx, plate_count in enumerate(plate_counts):
                cv.putText(
                    frame,
                    'Plate count: {}'.format(plate_count),
                    (40, 70 + 30 * idx),
                    font,
                    0.5,
                    (255, 255, 0),
                    2,
                    cv.LINE_AA)

            for idx, flag in enumerate(alarms_reached):
                if not flag and alarms[idx] == frame_counter:
                    alarms_reached[idx] = True

            for idx, alarm in enumerate(alarms):
                cv.putText(
                    frame,
                    'Alarm {}: {}'.format(idx, alarms_reached[idx]),
                    (40, 150 + 30 * idx),
                    font,
                    0.5,
                    (255, 0, 255),
                    2,
                    cv.LINE_AA)

            out.write(frame)
            frame_counter = frame_counter + 1

        out.release()
        cap.release()
