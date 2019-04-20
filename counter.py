import argparse
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from time import sleep
from scipy.signal import find_peaks

class Counter:

    def __init__(self, filename):
        self.filename = filename

    def analyse(self,
                    x_center, half_width,
                    y_center, half_height,
                    y_bar_start, bar_width, bar_height,
                    tracker_type='BOOSTING'):

        cap = cv.VideoCapture(self.filename)

        # Region of interest
        x_start = int(x_center - half_width)
        y_start = int(y_center - half_height)

        x_end = int(x_center + half_width)
        y_end = int(y_center + half_height)

        width = int(half_width * 2)
        height = int(half_height * 2)

        # Region of bar
        x_bar_start = int(half_width - bar_width / 2)
        x_bar_end = int(x_bar_start + bar_width)
        y_bar_end = int(y_bar_start + bar_height)

        # Window to track the bar
        bbox = (x_bar_start, y_bar_start, bar_width, bar_height)

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

        # Read first frame and extact the region of interest
        ok, frame = cap.read()
        frame = frame[y_start:y_end, x_start:x_end, :]

        # Define the bar to track
        ok = tracker.init(frame, bbox)

        # Draw bounding box of bar
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv.rectangle(frame, p1, p2, (255,0,0), 1, 1)

        # Resize for better visualization
        frame = cv.resize(frame, (width * 3, height * 3))

        # cv.imshow("Tracking", frame)
        sleep(0)

        y_pos_history = np.array(p1[1])

        while(1):
            sleep(0.0)
            ok, frame = cap.read()

            if not ok:
                break

            frame = frame[y_start:y_end, x_start:x_end, :]

            # Start timer
            timer = cv.getTickCount()

            ok, bbox = tracker.update(frame)

            # Calculate Frames per second (FPS)
            fps = cv.getTickFrequency() / (cv.getTickCount() - timer);

            # Draw bounding box
            if ok:
                # Tracking success
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                cv.rectangle(frame, p1, p2, (255,0,0), 1, 1)

                y_pos_history = np.hstack((y_pos_history, p1[1]))

            # Resize for better visualization
            frame = cv.resize(frame, (width * 3, height * 3))
            # cv.imshow("Tracking", frame)

            # Exit if ESC pressed
            k = cv.waitKey(1) & 0xff
            if k == 27:
                break

        # Normalize history
        min_ = np.min(y_pos_history)
        max_ = np.max(y_pos_history)

        y_pos_history = y_pos_history - min_
        y_pos_history = y_pos_history / (max_ - min_)

        plt.plot(y_pos_history)
        peaks, _ = find_peaks(y_pos_history, distance=15)

        valid_peaks = np.argwhere(y_pos_history[peaks] > 0.5)
        peaks = peaks[valid_peaks.ravel()]

        plt.plot(peaks, y_pos_history[peaks], 'X')

        plt.savefig('output')
        cv.destroyAllWindows()
        cap.release()

        print(peaks)
        self.draw_press_counter(peaks)

    def draw_press_counter(self, frame_numbers):
        cap = cv.VideoCapture(self.filename)
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        (major_ver, minor_ver, subminor_ver) = (cv.__version__).split('.')

        if int(major_ver)  < 3 :
            fps = cap.get(cv.cv.CV_CAP_PROP_FPS)
            print("Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps))
        else :
            fps = cap.get(cv.CAP_PROP_FPS)
            print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))


        out = cv.VideoWriter(
            "out.mp4",
            cv.VideoWriter_fourcc('H','2','6','4'),
            fps,
            (frame_width,frame_height))

        frame_counter = 0
        press_moves = 0
        font = cv.FONT_HERSHEY_SIMPLEX

        while(1):
            ok, frame = cap.read()

            if not ok:
                break

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



def main():
    parser = argparse.ArgumentParser(description='plates')
    parser.add_argument('filename', type=str, help='Video filename')

    args = parser.parse_args()

    counter = Counter(args.filename)
    counter.analyse(188, 40, 118, 24, 8, 35, 25)

if __name__ == '__main__':
    main()