import argparse
from copy import deepcopy
import os
import pprint
import typing

from skimage.measure import compare_ssim
import imutils
import cv2


class FileUtils:

    @classmethod
    def add_markup_to_filename(cls, name: str) -> str:
        """
        Appends "_markup" to the filename (maintains extension)
        Args:
            name: (str) Name of file

        Returns: (str) Name of file: [<path>]<filename>_markup.<ext>
        """
        parts = list(os.path.splitext(name))
        return "_markup".join(parts)


class CliArgs:
    COMPARE = 'compare'
    SOURCE = 'source'

    def __init__(self):
        self.ap = argparse.ArgumentParser()
        self.ap.add_argument(f"-{self.SOURCE[0].lower()}", f"--{self.SOURCE}", required=True,
                             help=f"{self.SOURCE.capitalize()} image input")
        self.ap.add_argument(f"-{self.COMPARE[0].lower()}", f"--{self.COMPARE}", required=True,
                             help=f"{self.COMPARE.capitalize()} image input")

    def get_args(self):
        return vars(self.ap.parse_args())


class Image:
    """
    Image processing Object.
    """
    def __init__(self, image_file_spec: str) -> None:
        """
        Constructor
        Args:
            image_file_spec: path/name of file (will be stored as an absolute path)
        """
        self.image_filespec = os.path.abspath(image_file_spec)
        self._image = None
        self._grayscale = None

        # Load image into memory
        if os.path.exists(self.image_filespec):
            self._image = cv2.imread(self.image_filespec)
        else:
            print(f"ERROR:\n\tImage file ({self.image_filespec}) not found.")

    def get_grayscale(self):
        """
        Retrieve the grayscale version.
            Lazy-loads image - does not generate until first call and returns results for all subsequent calls.

        Returns: Grayscale conversion of image
        """
        if self._grayscale is None:
            self._grayscale = cv2.cvtColor(self._image, cv2.COLOR_BGR2GRAY)
        return self._grayscale

    def get_image(self):
        """
        Returns the loaded image

        Returns: the loaded image (in memory)
        """
        return self._image

    def write_to_file(self, overwrite: bool = False, directory: typing.Optional[str] = None) -> bool:
        """
        Write image to disk, using the stored filename
        Args:
            overwrite: Boolean: If True, overwrite the image. Default = False
            directory: Str: file path to write file. Default = None

        Returns: Boolean: Image written to file
        """

        image_path = self.image_filespec
        image_path_written = False

        if directory is not None:
            filename = image_path.split(os.path.sep)[-1]
            image_path = os.path.abspath(os.path.sep.join([directory, filename]))

        # If no image path was specified
        if self.get_image() is None:
            print("ERROR: No image to save to file.")
            image_path_written = False

        elif self.image_filespec == '' or self.image_filespec is None:
            print("ERROR: No filename specified. Unable to save to file.")
            image_path_written = False

        # If image exists and overwrite is False
        elif os.path.exists(image_path) and not overwrite:
            print(f"WARNING: File ({image_path}) already exists. Not overwriting.")
            image_path_written = False

        # If image exists and overwrite is True, or image does not exist: write the file
        elif (os.path.exists(image_path) and overwrite) or not os.path.exists(image_path):
            cv2.imwrite(image_path, self.get_image())
            image_path_written = os.path.exists(image_path)
            if not image_path_written:
                print(f"ERROR: Unable to write {image_path} to file.")

        return image_path_written


class ImageCompare:
    """
    Compares two Image objects, using the Structural Similarities Index Method (SSIM).
        * Can identify specific difference locations (based on coordinates)
        * Can overlay a bounding rectangle around differences.
        * Can display differences
    """

    BGR_COLORS = {
        "RED": (0, 0, 255),
        "BLUE": (255, 0, 0),
        "CYAN": (255, 255, 0),
        "GREEN": (0, 255, 0),
        "MAGENTA": (255, 0, 255),
        "PURPLE": (128, 0, 128),
        "DARK_MAGENTA": (139, 0, 139),
    }

    BORDER_THICKNESS = 1

    def __init__(self, source_image: Image, compare_image: Image):
        """
        Constructor
        Args:
            source_image: Instantiated Image Object - Source of Truth (SoT) Image
            compare_image: Instantiated Image Object - Image to compare to SoT
        """
        self._source = source_image
        self._compare = compare_image
        self._source_markup = None
        self._compare_markup = None
        self._diff = None
        self._contours = None
        self._ssim_score = -100
        self._num_diffs = -1

    def get_source_image(self):
        return self._source

    def get_comparison_image(self):
        return self._compare

    def get_mark_up_images(self) -> typing.Tuple[Image, Image]:
        """
        Returns: Tuple of markup Image objs
        """
        return self._source_markup, self._compare_markup

    def get_ssim_score(self) -> float:
        """
        Get the SSIM score: range -1 to 1, where 1 = exact match; -100 - No score available.
        Returns: (float) [-1.0, 1,0]; -100
        """
        return self._ssim_score

    def get_number_of_diffs(self) -> int:
        """
        Returns the number of differences detected. -1 = Comparison not performed.

        Returns: Number of differences detected.
        """
        return len(self._contours) if self._contours is not None else -1

    def compare(self) -> "ImageCompare":
        """
        Compares images and determines contours where comparison image is different.

        Returns: self (allows method chaining).
        """
        (self._ssim_score, diff) = compare_ssim(self._source.get_grayscale(), self._compare.get_grayscale(), full=True)
        self._diff = (diff * 255).astype('uint8')

        thresh = cv2.threshold(self._diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        self._contours = imutils.grab_contours(
            cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE))

        return self

    def apply_differences(self) -> "ImageCompare":
        """
        Apply difference contours to image and create markup image in corresponding Image object.

        Returns: self (allows method chaining).
        """

        self._source_markup = deepcopy(self._source)
        self._source_markup.image_filespec = FileUtils.add_markup_to_filename(self._source_markup.image_filespec)
        self._compare_markup = deepcopy(self._compare)
        self._compare_markup.image_filespec = FileUtils.add_markup_to_filename(self._compare_markup.image_filespec)

        # Draw a bounding rectangle around each difference
        # Note: Color is defined as BGR versus the standard RGB. Why? Read link below...
        #      Reference: https://www.learnopencv.com/why-does-opencv-use-bgr-color-format/
        color_val_defs = list(self.BGR_COLORS.values())
        for color, contour in enumerate(self._contours):
            bgr_color = color_val_defs[color % len(color_val_defs)]
            (x, y, width, height) = cv2.boundingRect(contour)
            cv2.rectangle(
                self._source_markup.get_image(), (x, y), (x + width, y + height), bgr_color, self.BORDER_THICKNESS)
            cv2.rectangle(
                self._compare_markup.get_image(), (x, y), (x + width, y + height), bgr_color, self.BORDER_THICKNESS)

        return self

    def show_differences(self):
        """
        Display the source/comparison images and the corresponding markup images.

        Returns: None
        """
        cv2.imshow("Original Source Image", self._source.get_image())
        cv2.imshow("Original Compare Image", self._compare.get_image())

        cv2.imshow("Source Diff'd", self._source_markup.get_image())
        cv2.imshow("Compare Diff'd", self._compare_markup.get_image())

        cv2.waitKey(0)


class MultiDocCompare:
    """
    Compares a list of images, where each comparison is a tuple (src_filespec, cmp_filespec).
    Can generate a report of results.
    """
    def __init__(self, image_sets: typing.List[typing.Tuple[str, str]]):
        """
        Constructor
        Args:
            image_sets: List of tuples (src_filespec, cmp_filespec)
        """
        self.image_sets = image_sets
        self.comparisons = []
        self.report = []

    def compare_image_sets(self) -> None:
        """
        Performs comparison of each image tuple in the list.
        Returns: None
        """
        for index, image_set in enumerate(self.image_sets):
            comp_engine = ImageCompare(*[Image(image) for image in image_set])
            comp_engine.compare().apply_differences()
            show_comparison_stats(comp_engine)
            self.comparisons.append(comp_engine)

            for markup in comp_engine.get_mark_up_images():
                markup.write_to_file(overwrite=True, directory="./markup")

    def generate_report(self) -> typing.List[typing.List]:
        """
        Generates a list reporting list, where each element is a list of:
           * Source Filespec
           * Comparison Filespec
           * SSIM Score
           * Number of differences
           * Source Markup
           * Comparison Markup

        Returns: The list of lists report.
        """
        self.report = []
        for comp in self.comparisons:
            results = [comp.get_source_image().image_filespec, comp.get_comparison_image().image_filespec,
                       f"{comp.get_ssim_score():0.4}", comp.get_number_of_diffs()]
            results.extend([mup.image_filespec for mup in comp.get_mark_up_images()])
            self.report.append(results)
        return self.report


if __name__ == '__main__':
    def show_comparison_stats(image_compare):
        print(f"SSIM image difference score [-1, 1]: {image_compare.get_ssim_score():0.4}")
        print(f"Number of differences detected: {image_compare.get_number_of_diffs()}")

    SINGLE = False
    if SINGLE:
        args = CliArgs().get_args()

        src_image = Image(args[CliArgs.SOURCE])
        cmp_image = Image(args[CliArgs.COMPARE])

        comparison = ImageCompare(source_image=src_image, compare_image=cmp_image)
        comparison.compare().apply_differences()
        show_comparison_stats(comparison)

        for mark_up in comparison.get_mark_up_images():
            mark_up.write_to_file(overwrite=True, directory="./markup")

    else:
        images = [
            ('./picts/pict1.jpg', './picts/pict2.jpg'),
            ('./picts/link.jpg', './picts/link_tweaked.jpg'),
            ('./picts/1040.jpg', './picts/1040_updated.jpg')
        ]

        comps = MultiDocCompare(image_sets=images)
        comps.compare_image_sets()
        pprint.pprint(comps.generate_report())
