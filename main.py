import numpy as np
import argparse
from time import sleep, time
from counter import Counter
from press_counter import PressCounter
from plates_counter import PlatesCounter
from line_alarm import LineAlarm

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='plates')
    parser.add_argument('filename', type=str, help='Input video filename')
    parser.add_argument('-o', dest='outname', type=str,
        help='Name of the output video file.')

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

    counter = Counter(filename)
    press_counter = PressCounter(xc, hw, yc, hh, yb, bw, bh, tracker_type)

    xp1, yp1, wp1, hp1 = 137, 198, 48, 15
    xp2, yp2, wp2, hp2 = 216, 198, 42, 15
    plates_counter_1 = PlatesCounter(xp1, yp1, wp1, hp2)
    plates_counter_2 = PlatesCounter(xp2, yp2, wp2, hp2)

    p0 = [200, 296]
    p1 = [249, 295]
    p2 = [245, 474]
    p3 = [344, 465]
    x, y, w, h = 150, 100, 30, 180
    line_alarm = LineAlarm(np.array([p0, p1, p2, p3]), np.array([x, y, w, h]))

    start = time()
    counter.analyse(press_counter, [plates_counter_1, plates_counter_2], line_alarm, analysis)
    end = time()
    print("Time to process: {:d}s".format(int(end - start)))

    if outname is not None:
        # Draw text to coun the press moves in the video
        start = time()
        counter.draw_press_counter(outname, press_counter.peaks,
            [plates_counter_1.peaks, plates_counter_2.peaks], line_alarm.alarms)
        end = time()
        print("Time to create output video: {:d}s".format(int(end - start)))
