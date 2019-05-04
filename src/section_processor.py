import cv2 as cv
class SectionProcessor:
    """
    A class to process a section of frames.

    ...

    Methods
    -------
    init(frame)
        Initializes the processor.
    process_frame(frame)
        Process a frame.
    calculate_positions()
        Calculate positions of specific ocurrences.
    generate_report(fps, sub_name)
        Generates a report.
    draw_caption(value, frame, font, position=(0, 0), color=(0, 0, 255))
        Draws a caption in a frame.
    draw_text(text, frame, font, color, position)
        Draws text over a colored background in a frame.
    draw_processing_info(frame_number, frame, font, position=(0, 0), color=(0, 0, 255))
        Draws processing information of a frame.
    show_processing(frame, name="Processing")
        Displays the frame being processed.
    plot(name="Plot")
        Plots information of the processing done.
    calculate_events(fps, last_frame)
        Calculate important events in the processed video.
    state_at_frame(frame_number)
        Returns the state of the processor at a given frame.
    """

    def __init__(self):
        pass

    def init(self, frame):
        """
        Initializes the processor.

        Parameters
        ----------
        frame : ndarray
            3-channel image.
        """
        pass

    def process_frame(self, frame):
        """
        Processes a frame.

        Parameters
        ----------
        frame : ndarray
            3-channel image.
        """
        pass

    def calculate_positions(self):
        """
        Calculates positions of specific ocurrences.
        """
        pass

    def generate_report(self, fps, sub_name):
        """
        Generates a report.

        Parameters
        ----------
        fps : float
            Number of frames per second of the processed video.
        sub_name : str
            Partial name of the file report.
        """
        pass

    def draw_caption(self, value, frame, font, position=(0, 0), color=(0, 0, 255)):
        """
        Draws a caption in a frame.

        Parameters
        ----------
        value : any
            Value to be drawn in a frame as text.
        frame : ndarray
            3-channel image.
        font : int
            Font type available in OpenCV.
        position : (int, int)
            Position in the frame to start drawing the caption.
        color : (int, int, int)
            BGR color of the caption.
        """
        pass

    def draw_text(self, text, frame, font, color, position):
        """
        Draws text over a colored background in a frame.

        Parameters
        ----------
        text : str
            Text to be drawn in the frame
        frame : ndarray
            3-channel image.
        font : int
            Font type available in OpenCV.
        color : (int, int, int)
            BGR color of the text.
        position : (int, int)
            Position in the frame to start drawing the caption.
        """
        font_scale = 0.3
        thickness = 1

        box_coords = self.__get_box_text(position, text, font, font_scale, thickness)

        cv.rectangle(frame, box_coords[0], box_coords[1], (255, 255, 255), cv.FILLED)
        cv.putText(
            frame,
            text,
            position,
            font,
            font_scale,
            color,
            thickness,
            cv.LINE_AA)

    def __get_box_text(self, position, text, font, font_scale, thickness):
        """
        Calculates the bounding box of a text with a small padding

        Parameters
        ---------
        position : (int, int)
            Position in the frame where the text will be drawn.
        text : str
            Text to be drawn in a frame.
        font : int
            Font type available in OpenCV.
        font_scale : float
            Size of the text.
        thickness: int
            Thickness of the text.

        Returns
        -------
        ((int, int), (int, (int))) : Coordinates of top-left and bottom-right
        corners of the bounding box.
        """
        (text_width, text_height) = cv.getTextSize(text, font, font_scale, thickness)[0]
        box_coords = (
            position,
            (position[0] + text_width - 2, position[1] - text_height - 2))

        return box_coords

    def draw_processing_info(self, frame_number, frame, font, position=(0, 0), color=(0, 0, 255)):
        """
        Draws processing information of a frame.

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
        pass

    def show_processing(self, frame, name="Processing"):
        """
        Displays the frame being processed.

        Parameters
        ----------
        frame : ndarray
            3-channel image.
        name : str
            Name of the window to display the frame.
        """
        pass

    def plot(self, name="Plot"):
        """
        Plots information of the processing done.

        Parameters
        ----------
        name : str
            Name of the plot.
        """
        pass

    def calculate_events(self, fps, last_frame):
        """
        Calculate important events in the processed video.

        Parameters
        ----------
        fps : float
            Frames per second of the processed video.
        last_frame : int
            Index of the last frame in the processed video
        """
        pass

    def state_at_frame(self, frame_number):
        """
        Returns the state of the processor at a given frame.

        Parameters
        ----------
        frame_number : int
            The point to get the state.
        """
        pass