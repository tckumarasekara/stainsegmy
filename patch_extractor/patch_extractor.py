import math
import numpy as np


class PatchExtractor(object):
    """
    Extract patches from the input image. The patch size and step size are defined by the user.
    """

    def __init__(self, image, win_size, step_size):
        """
        Initialize the PatchExtractor object.

        Args:
        - image (numpy array): The input image from which patches will be extracted. It should be of shape (H, W, C).
        - win_size (int): The size of the patch to be extracted.
        - step_size (int): The step size for sliding the window to extract patches.
        """

        assert isinstance(win_size, int)
        assert isinstance(step_size, int)

        self.patch_type = "mirror"
        self.win_size = [win_size, win_size]
        self.step_size = [step_size, step_size]
        self.counter = 0
        self.__pad_if_needed(image)


    def __pad_if_needed(self, x):
        """
        Pad the input image if its dimensions are smaller than the specified window size or if the remaining portion 
        after sliding the window does not fit perfectly. The padding is done using reflection to ensure that the 
        central region of each patch is always within the original image.

        Args:
        - x (numpy array): The input image to be padded if needed.

        Returns:
        - None
        """
            
        img_y, img_x = x.shape[:2]

        if img_x < self.win_size[1]:
            diff_x = self.win_size[1] - img_x
            x = np.lib.pad(x, ((0, 0), (0, diff_x), (0, 0)), "reflect")
        if img_y < self.win_size[0]:
            diff_y = self.win_size[0] - img_y
            x = np.lib.pad(x, ((0, diff_y), (0, 0), (0, 0)), "reflect")

        if (img_x - self.win_size[1]) % self.step_size[1] != 0:
            diff_x = self.step_size[1] - (img_x - self.win_size[1]) % self.step_size[1]
            x = np.lib.pad(x, ((0, 0), (0, diff_x), (0, 0)), "reflect")
        
        if (img_y - self.win_size[0]) % self.step_size[0] != 0:
            diff_y = self.step_size[0] - (img_y - self.win_size[0]) % self.step_size[0]
            x = np.lib.pad(x, ((0, diff_y), (0, 0), (0, 0)), "reflect")

        self.image = x


    def extract_infos(self, length, win_size, step_size):
        """
        Extract information about the sliding window parameters.

        Args:
        - length (int): The length of the dimension along which patches are extracted.
        - win_size (int): The size of the patch to be extracted.
        - step_size (int): The step size for sliding the window to extract patches.

        Returns:
        - int: The last step position for patch extraction.
        """
            
        last_step = math.floor((length - win_size) / step_size)
        last_step = 0 if last_step == 0 else (last_step + 1) * step_size
        
        return last_step


    def get_patch(self, ptx):
        """
        Get a patch from the input image based on the provided top-left corner coordinates.

        Args:
        - ptx (tuple): The top-left corner coordinates of the patch to be extracted.

        Returns:
        - numpy array: The extracted patch.
        """
        pty = (ptx[0] + self.win_size[0], ptx[1] + self.win_size[1])
        patch = self.image[ptx[0]: pty[0], ptx[1]: pty[1]]

        return patch



class MaskStitcher(object):
    """
    Stitch the predicted patches back together to form the final segmentation mask. The stitching is done in a way 
    that ensures the central region of each patch is always within the original image, and the edges are handled
    """

    def __init__(self, im_h, im_w, win_size, step_size):
        """
        Initialize the MaskStitcher object.

        Args:
        - im_h (int): The height of the original image.
        - im_w (int): The width of the original image.
        - win_size (int): The size of the patch that was extracted.
        - step_size (int): The step size that was used for sliding the window to extract patches.
        """
        self.im_h = im_h
        self.im_w = im_w
        self.win_size = (win_size, win_size)
        self.step_size = (step_size, step_size)
        self.mask = self.__pad_if_needed(np.zeros((im_h, im_w), dtype=np.uint8))


    def __pad_if_needed(self, x):
        """
        Pad the input mask if its dimensions are smaller than the specified window size or if the remaining portion 
        after sliding the window does not fit perfectly. The padding is done using reflection to ensure that the 
        central region of each patch is always within the original image.

        Args:
        - x (numpy array): The input mask to be padded if needed.

        Returns:
        - numpy array: The padded mask if padding was needed, otherwise the original mask.
        """

        img_y, img_x = x.shape[:2]

        if img_x < self.win_size[1]:
            diff_x = self.win_size[1] - img_x
            x = np.lib.pad(x, ((0, 0), (0, diff_x)), "reflect")

        if img_y < self.win_size[0]:
            diff_y = self.win_size[0] - img_y
            x = np.lib.pad(x, ((0, diff_y), (0, 0)), "reflect")

        if (img_x - self.win_size[1]) % self.step_size[1] != 0:
            diff_x = self.step_size[1] - (img_x - self.win_size[1]) % self.step_size[1]
            x = np.lib.pad(x, ((0, 0), (0, diff_x)), "reflect")

        if (img_y - self.win_size[0]) % self.step_size[0] != 0:
            diff_y = self.step_size[0] - (img_y - self.win_size[0]) % self.step_size[0]
            x = np.lib.pad(x, ((0, diff_y), (0, 0)), "reflect")

        return x


    def stitch(self, patch, ptx, end_x=False, end_y=False):
        """
        Stitch the predicted patch back into the final segmentation mask based on the provided top-left corner coordinates. 
        The stitching is done in a way that ensures the central region of each patch is always within the original image, 
        and the edges are handled appropriately based on the specified edge cases.

        Args:
        - patch (numpy array): The predicted patch to be stitched back into the final mask.
        - ptx (tuple): The top-left corner coordinates of the patch to be stitched.
        - end_x (str, optional): The edge case for the x-dimension. It can be "left", "right", "both", or False (default). 
                                If "left", the left edge of the patch is at the edge of the image. 
                                If "right", the right edge of the patch is at the edge of the image.
                                If "both", both edges of the patch are at the edge of the image.
        - end_y (str, optional): The edge case for the y-dimension. It can be "top", "bottom", "both", or False (default). 
                                If "top", the top edge of the patch is at the edge of the image. 
                                If "bottom", the bottom edge of the patch is at the edge of the image.
                                If "both", both edges of the patch are at the edge of the image.

        Returns:
        - None
        """

        x0, y0 = ptx
        x1, y1 = x0 + self.win_size[0], y0 + self.win_size[1]

        crop = 25

        top = crop
        bottom = crop
        left = crop
        right = crop

        # edge case, crop only the side that is not at the edge
        if end_x == "left":
            left = 0
        elif end_x == "right":
            right = 0
        elif end_x == "both":
            left = right = 0

        if end_y == "top":
            top = 0
        elif end_y == "bottom":
            bottom = 0
        elif end_y == "both":
            top = bottom = 0

        self.mask[x0 + top : x1 - bottom, y0 + left : y1 - right] = patch[top : patch.shape[0] - bottom, left : patch.shape[1] - right]


    def get_mask(self):
        """
        Get the final stitched segmentation mask. The mask is cropped to the original image dimensions to remove any padding 
        that was added during patch extraction.

        Returns:
        - numpy array: The final stitched segmentation mask, cropped to the original image dimensions.
        """
        
        self.mask = self.mask[: self.im_h, : self.im_w]
        return self.mask



if __name__ == "__main__":
    # example for debug
    xtractor = PatchExtractor((256, 256), (128, 128), debug=True)
    a = np.full([1200, 1200, 3], 255, np.uint8)
    xtractor.extract(a, "mirror")
    xtractor.extract(a, "valid")
