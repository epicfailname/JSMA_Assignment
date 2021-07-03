import os

import cv2 as cv
import imageio as im
import matplotlib.pyplot as plt
import numpy as np
import skimage.transform as st
from skimage.draw import polygon2mask
from sklearn.utils import shuffle

IDENTITY_FN = lambda x: x

# Converts row-major vertice to col-major point
def pixel_to_point(i):
    p = np.array([i[1], i[0]])
    return p

# Converts col-major point to row-major vertice
def point_to_pixel(p):
    i = np.array([p[1], p[0]])
    return i

def get_homography(from_pixel, to_pixel):
    from_points = np.asarray([pixel_to_point(i) for i in from_pixel])
    to_points = np.asarray([pixel_to_point(i) for i in to_pixel])
    hom, _ = cv.findHomography(from_points, to_points, cv.RANSAC, 5.0)
    return hom

def dot_homography(hom, from_pixel):
    from_point = pixel_to_point(from_pixel)
    from_point_3d = np.array([from_point[0], from_point[1], 1])
    from_point_3d.reshape([3, 1])
    res = np.dot(hom, from_point_3d)
    res = res / res[2]
    return point_to_pixel([round(res[0]), round(res[1])])

def generate_hull_mask(shape, hull_corners):
    mask = polygon2mask(shape, np.array(hull_corners))
    return mask

def parse_corners_string(corners_string):
    corners = [[int(i) for i in r.split(' ')] for r in corners_string.strip().split('|')]
    return corners

def plot_and_save_graph(list_of_data, title, xlabel, ylabel, savedir):
    number_of_lines = len(list_of_data)
    colormap = plt.cm.brg
    colors = [colormap(i) for i in np.linspace(0, 1, number_of_lines)]
    
    fig, ax = plt.subplots()
    for i in range(number_of_lines):
        ax.plot(list_of_data[i], label = 'Frame ' + str(i))
    for i, j in enumerate(ax.lines):
        j.set_color(colors[i])
    ax.legend(loc='center left', bbox_to_anchor=(1.05, 0.5), fancybox=True, shadow=True)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.savefig(os.path.join(savedir, title + '.png'), bbox_inches = 'tight')
    plt.close(fig)

def read_center_images_angles(steering_image_log, image_folder, transform_img_fn, is_shuffle=True):
    steerings = []
    img_filenames = []
    imgs = []

    # Read steering angles and image filenames
    with open(steering_image_log) as f:
        for line in f.readlines()[1:]:
            if 'center' not in line:
                continue

            fields = line.split(",")
            steering = float(fields[6])
            url = os.path.join(image_folder, fields[5])

            steerings.append(steering)
            img_filenames.append(url)

    # Shuffle in parallel
    if is_shuffle:
        img_filenames, steerings = shuffle(img_filenames, steerings)

    # Read processed images into memory
    for url in img_filenames:
        img = im.imread(url)
        img = transform_img_fn(img)

        imgs.append(img)

    return img_filenames, imgs, steerings

def read_center_images_angles_corners(steering_image_log,
        image_folder,
        transform_img_fn,
        transform_pixel_fn,
        is_shuffle=False):
    steerings = []
    img_filenames = []
    imgs = []
    sequence_of_list_of_corners = []
    raw_sequence_of_list_of_corners = []

    # Read steering angles and image filenames
    with open(steering_image_log) as f:
        for line in f.readlines()[1:]:
            if 'center' not in line:
                continue

            fields = line.split(",")
            steering = float(fields[6])
            url = os.path.join(image_folder, fields[5])
            corners_of_mask = None if fields[12] in (None, '\n', '') else parse_corners_string(fields[12])

            steerings.append(steering)
            img_filenames.append(url)
            sequence_of_list_of_corners.append(corners_of_mask)
    raw_sequence_of_list_of_corners = sequence_of_list_of_corners.copy()

    # Shuffle in parallel
    if is_shuffle:
        img_filenames, steerings, sequence_of_list_of_corners = shuffle(img_filenames, steerings, sequence_of_list_of_corners)

    img = im.imread(os.path.join(image_folder, img_filenames[0]))
    raw_img_shape = img.shape

    # Read processed images into memory
    for i in range(len(img_filenames)):
        url = img_filenames[i]
        img = im.imread(url)

        transformed_img = transform_img_fn(img)
        imgs.append(transformed_img)

        # Transform the vertices of masks
        sequence_of_list_of_corners[i] = None if sequence_of_list_of_corners[i] == None else [transform_pixel_fn(c, raw_img_shape[0:2]) for c in sequence_of_list_of_corners[i]]

    for list_of_corners in sequence_of_list_of_corners:
        if list_of_corners == None:
            continue
        for corner in list_of_corners:
            for elem in corner:
                assert(elem >= 0)

    return img_filenames, imgs, steerings, sequence_of_list_of_corners, raw_sequence_of_list_of_corners, raw_img_shape

class SDC_Nvidia_data:
    static_resize_shape = (128, 128)

    @staticmethod
    def static_resize_img_fn(img):
        return st.resize(img, SDC_Nvidia_data.static_resize_shape)
    @staticmethod
    def static_resize_pixel_row_fn(pixel_row, img_height): 
        return round(pixel_row / img_height * SDC_Nvidia_data.static_resize_shape[0])
    @staticmethod
    def static_resize_pixel_col_fn(pixel_col, img_width):
        return round(pixel_col / img_width * SDC_Nvidia_data.static_resize_shape[1])
    @staticmethod
    def static_resize_pixel_fn(pixel, img_shape):
        return [
                SDC_Nvidia_data.static_resize_pixel_row_fn(pixel[0], img_shape[0]),
                SDC_Nvidia_data.static_resize_pixel_col_fn(pixel[1], img_shape[1])]

    @staticmethod
    def static_crop_img_fn(img):
        return img[100:,20:-20,:]
    @staticmethod
    def static_crop_pixel_row_fn(pixel_row, img_height):
        return pixel_row - 100
    @staticmethod
    def static_crop_pixel_col_fn(pixel_col, img_width):
        return pixel_col - 20
    @staticmethod
    def static_crop_pixel_fn(pixel, img_shape):
        return [
                SDC_Nvidia_data.static_crop_pixel_row_fn(pixel[0], img_shape[0]),
                SDC_Nvidia_data.static_crop_pixel_col_fn(pixel[1], img_shape[1])]
    @staticmethod
    def static_crop_shape_fn(shape):
        return [shape[0] - 100, shape[1] - 40]

    def __init__(self,
            image_file,
            image_folder,
            is_shuffle=False,
            is_crop=True,
            is_mask=True):

        if is_crop:
            crop_img_fn = self.static_crop_img_fn  
            crop_pixel_fn = self.static_crop_pixel_fn
        else:
            crop_img_fn = IDENTITY_FN
            crop_pixel_fn = IDENTITY_FN

        self.transform_img_fn = lambda img: self.static_resize_img_fn(crop_img_fn(img))
        self.transform_pixel_fn = lambda pixel, img_shape: self.static_resize_pixel_fn(crop_pixel_fn(pixel, img_shape), self.static_crop_shape_fn(img_shape))

        if not is_mask:
            filenames, data, labels = read_center_images_angles(
                    image_file,
                    image_folder,
                    transform_img_fn = self.transform_img_fn,
                    is_shuffle = is_shuffle)
            self.input_data_filenames = np.asarray(filenames)
            self.input_data = np.asarray(data)
            self.output_data = np.asarray(labels)
        else:
            filenames, data, labels, sequence_of_list_of_corners, raw_sequence_of_list_of_corners, raw_data_shape = read_center_images_angles_corners(
                    image_file,
                    image_folder,
                    transform_img_fn = self.transform_img_fn,
                    transform_pixel_fn = self.transform_pixel_fn,
                    is_shuffle = is_shuffle)
            self.input_data_filenames = np.asarray(filenames)
            self.input_data = np.asarray(data)
            self.output_data = np.asarray(labels)
            self.sequence_of_list_of_corners = np.asarray(sequence_of_list_of_corners)
            self.raw_input_data_shape = raw_data_shape
            self.raw_sequence_of_list_of_corners = np.asarray(raw_sequence_of_list_of_corners)

    def __len__(self) :
        return (np.ceil(len(self.input_data) / float(self.batch_size))).astype(np.int)

class SDC_data:
    static_resize_shape = (128, 128)

    @staticmethod
    def static_resize_img_fn(img):
        return st.resize(img, SDC_data.static_resize_shape)
    @staticmethod
    def static_resize_pixel_row_fn(pixel_row, img_height): 
        return round(pixel_row / img_height * SDC_data.static_resize_shape[0])
    @staticmethod
    def static_resize_pixel_col_fn(pixel_col, img_width):
        return round(pixel_col / img_width * SDC_data.static_resize_shape[1])
    @staticmethod
    def static_resize_pixel_fn(pixel, img_shape):
        return [SDC_data.static_resize_pixel_row_fn(pixel[0], img_shape[0]),
                SDC_data.static_resize_pixel_col_fn(pixel[1], img_shape[1])]

    @staticmethod
    def static_crop_img_fn(img):
        return img[200:,:]
    @staticmethod
    def static_crop_pixel_row_fn(pixel_row, img_height):
        return pixel_row - 200
    @staticmethod
    def static_crop_pixel_col_fn(pixel_col, img_width):
        return pixel_col
    @staticmethod
    def static_crop_pixel_fn(pixel, img_shape):
        return [SDC_data.static_crop_pixel_row_fn(pixel[0], img_shape[0]),
                SDC_data.static_crop_pixel_col_fn(pixel[1], img_shape[1])]

    def __init__(self,
            image_file,
            image_folder,
            is_shuffle=False,
            is_crop=True,
            is_mask=True):

        if is_crop:
            crop_img_fn = self.static_crop_img_fn  
            crop_pixel_fn = self.static_crop_pixel_fn
        else:
            crop_img_fn = IDENTITY_FN
            crop_pixel_fn = IDENTITY_FN

        self.transform_img_fn = lambda img: self.static_resize_img_fn(crop_img_fn(img))
        self.transform_pixel_fn = lambda pixel, img_shape: self.static_resize_pixel_fn(crop_pixel_fn(pixel, img_shape), crop_pixel_fn(img_shape, img_shape))

        if not is_mask:
            filenames, data, labels = read_center_images_angles(image_file,
                    image_folder,
                    transform_img_fn = self.transform_img_fn,
                    is_shuffle = is_shuffle)
            self.input_data_filenames = np.asarray(filenames)
            self.input_data = np.asarray(data)
            self.output_data = np.asarray(labels)
        else:
            filenames, data, labels, sequence_of_list_of_corners, raw_sequence_of_list_of_corners, raw_data_shape = read_center_images_angles_corners(image_file,
                    image_folder,
                    transform_img_fn = self.transform_img_fn,
                    transform_pixel_fn = self.transform_pixel_fn,
                    is_shuffle = is_shuffle)
            self.input_data_filenames = np.asarray(filenames)
            self.input_data = np.asarray(data)
            self.output_data = np.asarray(labels)
            self.sequence_of_list_of_corners = np.asarray(sequence_of_list_of_corners)
            self.raw_input_data_shape = raw_data_shape
            self.raw_sequence_of_list_of_corners = np.asarray(raw_sequence_of_list_of_corners)

    def __len__(self) :
        return len(self.input_data)
