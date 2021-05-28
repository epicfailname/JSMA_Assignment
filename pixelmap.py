from enum import Enum

import cv2 as cv
import numpy as np
import tensorflow as tf

from utilities import (dot_homography, generate_hull_mask, get_homography,
                       pixel_to_point, point_to_pixel)

class AlgorithmEnum(Enum):
    HOMOGRAPHY = 1

"""
    Terminology:
    vertice:
        Row-major indexing. Includes the RGB axis. 
        shape: [HEIGHT, WIDTH, 3]
    pixel:
        Row-major indexing. Does not include RGB axis.
        shape: [HEIGHT, WIDTH]
    corner:
        A particular case of an pixel.
        A "vertice" of a 2-dimensional polygon (row-major indexing). Does not include RGB axis.
        shape: [HEIGHT, WIDTH]
    point:
        Column-major indexing of a corner. Does not include RGB axis.
        shape: [WIDTH, HEIGHT]
    list_of_*:
        General iterable. Assume numpy:array type.
    sequence_of_*:
        Refers to an iterable where axis 0 indexes the frames/images.
        E.g. sequence_of_list_of_corners:
            Is the iterable where [i][j] indexes the i-th frame, and j-th corner of the i-th frame.
    vertice_string(s):
        Refers to an iterable where axis 0 indexes the frames/images, and
        each element in the iterable are the same pixel, but possibly transposed.
        This is a chained-mapping from the referance (last) frame vertice to each previous frame vertice.
        E.g. list_of_vertice_strings:
            Is the iterable where [i][j] indexes the i-th vertice_string, and j-th frame's vertice in the mapping.
"""

class PixelMap:

    def __init__(self, list_of_vertice_strings, frame_shape):
        self.list_of_vertice_strings = list_of_vertice_strings
        self.frame_shape = frame_shape

    def get_list_of_vertice_strings(self):
        return self.list_of_vertice_strings

    def get_length_of_vertice_strings(self):
        return self.list_of_vertice_strings.shape[0]

    def delete_vertice_string(self, indice):
        self.list_of_vertice_strings = np.delete(self.list_of_vertice_strings, indice, axis = 0)

class Homography(PixelMap):

    def __init__(self, sequence_of_list_of_corners, frame_shape):
        num_frames = len(sequence_of_list_of_corners)

        # Using the last frame as the reference frame
        ref_frame_list_of_corners = sequence_of_list_of_corners[num_frames - 1]
        ref_frame_mask = generate_hull_mask(frame_shape, np.array(ref_frame_list_of_corners))

        # Get pixel of the last frame which we want the vertice mapping
        # (vertice_strings) for
        # These vertices are (HEIGHT, WIDTH), no RGB index.
        ref_frame_list_of_pixels = tf.sparse.from_dense(tf.where(ref_frame_mask, x = 1, y = 0)).indices

        # Calculate the homography matrix from the reference frame to each of
        # the other frames
        homs = []

        for b in range(num_frames - 1):
            homs.append(get_homography(sequence_of_list_of_corners[num_frames - 1], sequence_of_list_of_corners[b]))

        list_of_vertice_strings = []
        for pixel in ref_frame_list_of_pixels:
            # Since we want the RGB indexes, we need to duplicate the
            # vertice_strings three times.
            vertice_strings = [[], [], []]

            # For each frame other than the reference frame
            for b in range(num_frames - 1):
                # Find corresponding mapped vertice
                mapped_pixel = dot_homography(homs[b], pixel)
                # And append the coordinate (extended with batch and RGB
                # dimensions)
                for c in range(3):
                    t = np.concatenate([mapped_pixel, [c]])
                    t = np.concatenate([[b], t])
                    # t shape: [FRAME, HEIGHT, WIDTH, RGB]
                    vertice_strings[c].append(t)
            # Then append the coordinate of the reference frame
            for c in range(3):
                t = np.concatenate([pixel, [c]])
                t = np.concatenate([[num_frames - 1], t])
                vertice_strings[c].append(t)
            list_of_vertice_strings.extend(np.asarray(vertice_strings, dtype=np.int32))

        super().__init__(np.asarray(list_of_vertice_strings), frame_shape)
