import numpy as np
import argparse
from time import sleep, time
from counter import Counter
from press_counter import PressCounter
from plates_counter import PlatesCounter
from line_alarm import LineAlarm
from utils import readable_time, string_to_bool, unreadable_time

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='plates')
    parser.add_argument('filename', type=str, help='Input video filename')

    # Options for the press.
    parser.add_argument('-pxc', dest='press_x_center', type=int, default=188,
        help='Horizontal position of the center of the main area of the press.')

    parser.add_argument('-phw', dest='press_half_width', type=int, default=40,
        help='Half width of the main area of the press.')

    parser.add_argument('-pyc', dest='press_y_center', type=int, default=118,
        help='Vertical position of the center of the main area of the press.')

    parser.add_argument('-phh', dest='press_half_height', type=int, default=24,
        help='Half height of the main area of the press.')

    parser.add_argument('-pyb', dest='press_y_bar_start', type=int, default=8,
        help='Vertical position of the inner area of the press. This value is relative to the main area.')

    parser.add_argument('-pbw', dest='press_half_bar_width', type=int, default=17,
        help='Half width of the inner area of the press.')

    parser.add_argument('-pbh', dest='press_half_bar_height', type=int, default=12,
        help='Half height of the inner area of the press.')

    parser.add_argument('-ptt', dest='press_tracker_type', type=str, default='BOOSTING',
        help='Name of one of the built-in trackers in OpenCV.')

    # Options for the shoots.
    parser.add_argument('-sx1', dest='shoot_x_1', type=int, default=137,
        help='Horizontal position of top corner of left shoot.')

    parser.add_argument('-sy1', dest='shoot_y_1', type=int, default=198,
        help='Vertical position of top corner of left shoot.')

    parser.add_argument('-sw1', dest='shoot_w_1', type=int, default=48,
        help='Width of left shoot.')

    parser.add_argument('-sh1', dest='shoot_h_1', type=int, default=15,
        help='Height of left shoot.')

    parser.add_argument('-sx2', dest='shoot_x_2', type=int, default=216,
        help='Horizontal position of top corner of right shoot.')

    parser.add_argument('-sy2', dest='shoot_y_2', type=int, default=198,
        help='Vertical position of top corner of right shoot.')

    parser.add_argument('-sw2', dest='shoot_w_2', type=int, default=42,
        help='Width of right shoot.')

    parser.add_argument('-sh2', dest='shoot_h_2', type=int, default=15,
        help='Height of right shoot.')

    #Options for the bands.
    parser.add_argument('-bx1', dest='upper_left_x', type=int, default=200,
        help='Horizontal position of upper mark in left band.')

    parser.add_argument('-by1', dest='upper_left_y', type=int, default=296,
        help='Vertical position of upper mark in left band.')

    parser.add_argument('-bx2', dest='upper_right_x', type=int, default=249,
        help='Horizontal position of upper mark in right band.')

    parser.add_argument('-by2', dest='upper_right_y', type=int, default=295,
        help='Vertical position of upper mark in right band.')

    parser.add_argument('-bx3', dest='lower_left_x', type=int, default=245,
        help='Horizontal position of lower mark in left band.')

    parser.add_argument('-by3', dest='lower_left_y', type=int, default=474,
        help='Vertical position of lower mark in left band.')

    parser.add_argument('-bx4', dest='lower_right_x', type=int, default=344,
        help='Horizontal position of lower mark in right band.')

    parser.add_argument('-by4', dest='lower_right_y', type=int, default=465,
        help='Vertical position of lower mark in right band.')

    parser.add_argument('-bxd', dest='box_x', type=int, default=150,
        help='Horizontal position of upper left corner of destination box.')

    parser.add_argument('-byd', dest='box_y', type=int, default=100,
        help='Vertical position of upper left corner of destination box.')

    parser.add_argument('-bwd', dest='box_w', type=int, default=30,
        help='Width of destination box.')

    parser.add_argument('-bhd', dest='box_h', type=int, default=180,
        help='Height of destination box.')

    # Extra options
    parser.add_argument('-o', dest='outname', type=str,
        help='Name of the output video file.')

    parser.add_argument('-a', dest='analysis', type=string_to_bool, default='0',
        help='Flag used for analysis.')

    parser.add_argument('-from', dest='from_', type=unreadable_time, default='00:00:00',
        help="Start point to process the video.")

    parser.add_argument('-to', dest='to_', type=unreadable_time, default='end',
        help="End point to process the video.")

    args = parser.parse_args()

    filename = args.filename
    outname = args.outname

    # PressCounter
    xc = args.press_x_center
    hw = args.press_half_width
    yc = args.press_y_center
    hh = args.press_half_height
    yb = args.press_y_bar_start
    bw = args.press_half_bar_width
    bh = args.press_half_bar_height
    tracker_type = args.press_tracker_type

    press_counter = PressCounter(xc, hw, yc, hh, yb, bw, bh, tracker_type)

    # PlatesCounter
    xp1, xp2 = args.shoot_x_1, args.shoot_x_2
    yp1, yp2 = args.shoot_y_1, args.shoot_y_2
    wp1, wp2 = args.shoot_w_1, args.shoot_w_2
    hp1, hp2 = args.shoot_h_1, args.shoot_h_2

    plates_counter_1 = PlatesCounter(xp1, yp1, wp1, hp1)
    plates_counter_2 = PlatesCounter(xp2, yp2, wp2, hp2)

    # LineAlarm
    p0 = [args.upper_left_x  , args.upper_left_y  ]
    p1 = [args.upper_right_x , args.upper_right_y ]
    p2 = [args.lower_left_x  , args.lower_left_y  ]
    p3 = [args.lower_right_x , args.lower_right_y ]
    x, y, w, h = args.box_x, args.box_y, args.box_w, args.box_y

    line_alarm = LineAlarm(np.array([p0, p1, p2, p3]), np.array([x, y, w, h]))

    # Counter
    analysis = args.analysis
    start_time = args.from_
    end_time = args.to_
    processors = [plates_counter_1, plates_counter_2, press_counter, line_alarm]

    counter = Counter(filename, processors, (start_time, end_time))

    start = time()
    counter.analyse(analysis)
    end = time()
    print("Time to process: {}".format(readable_time(int(end - start))))

    counter.generate_report()

    if outname is not None:
        # Draw text to coun the press moves in the video
        start = time()
        counter.create_output_video(outname)
        end = time()
        print("Time to create output video: {}".format(readable_time(int(end - start))))
