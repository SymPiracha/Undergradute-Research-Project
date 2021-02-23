#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 21:32:04 2020

@author: Jiuqi Wang
"""

import glob
import cv2
import numpy as np
import math
import imagehash
from PIL import Image
from shapely.geometry import LineString
import itertools

"""
input: scene_name as a String specifying the class of images , folder as a String specifying the path, 
       set_height as an integer to resize all the images being loaded
output: a list of resized image(matrix) in grayscale
"""


def load_img(scene_name="Hallway", folder="../dataset/", set_height=500):
    # load the images from the folder as a list of gray scale images
    images = [cv2.imread(file, cv2.IMREAD_GRAYSCALE) for file in glob.glob(folder + scene_name + "*.jpg")]
    for i in range(len(images)):
        img = images[i]
        # get the height and width of each image
        height, width = img.shape
        # calculate the scale factor to resize the images to the desired sizes
        scale_factor = set_height / height
        # calculate the resized height and width
        resized_width = int(scale_factor * width)
        resized_dim = (resized_width, set_height)
        # resize the images
        images[i] = cv2.resize(img, resized_dim, interpolation=cv2.INTER_AREA)
    return images


"""
input: image as a matrix, img_name as a String specifying the name of the image
output: None
This function displays the input image in another window. Press any key to terminate the window
"""


def show_img(image, img_name="Hallway"):
    cv2.imshow(img_name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.waitKey(1)


'''
input: image as a matrix, window_size determines the window size of the gaussian filters,
       sigma as the standard deviation, threshold1 and thresholds specify the two
       thresholds for the canny edge detector
output: a binary image that contains the thickened edges of the input image
'''


def detect_edge(image, window_size=9, sigma=2, threshold1=20, threshold2=60):
    # smooth the image
    smoothed = cv2.GaussianBlur(image, (window_size, window_size), sigma)
    # detect edges
    edges_detected = cv2.Canny(smoothed, threshold1, threshold2)

    return edges_detected


"""
input: y and x as the rise and run used to calculate arctan(y/x)
output: angle formed by y and x in [0, 360)
"""


def calculate_angle(y, x):
    angle_in_rad = math.atan2(y, x)
    angle_in_deg = (angle_in_rad * 180 / math.pi + 360) % 360
    return angle_in_deg


"""
input: image_edges as a matrix, min_length specifying minimum acceptable line length, max_gap specifying the 
       maximum allowable gap 
output: lines in matrix form. Each row is comprised of x1, y1, x2, y2, angle.
"""


def detect_lines(image_edges, min_length=40, max_gap=60):
    detected_lines = cv2.HoughLinesP(image_edges, 1, np.pi / 180, 70, minLineLength=min_length, maxLineGap=max_gap)
    # reshape the matrix
    detected_lines = detected_lines.reshape(-1, 4)
    num_of_lines, _ = detected_lines.shape
    # add an extra column
    complete = np.zeros((num_of_lines, 5))
    # calculate angle
    for index, line in enumerate(detected_lines):
        x1, y1, x2, y2 = line
        complete[index, 0:4] = line
        complete[index, 4] = calculate_angle(y2 - y1, x2 - x1)

    return complete


"""
input: lines as an 2D array, with each row representing a line; size_of_pic as a tuple (height, width)
output: a 2D dimensional array with all lines starts and ends on the border of the picture

"""


def extend_lines(lines, size_of_pic):
    extended_lines = [extend_one(line, size_of_pic) for line in lines]
    return extended_lines


"""
helper function of extend_lines that extends only one single line

"""


def extend_one(line, size_of_pic):
    if line[4] == 90 or line[4] == 270:  # when line is vertical
        return [line[0], 0, line[0], size_of_pic[0], line[4]]

    slope = math.tan(line[4] * math.pi / 180)
    b = line[1] - slope * line[0]

    # extended end points
    if slope == 0:  # when line horizontal
        return [0, line[1], size_of_pic[1], line[1], line[4]]
    if slope > 0:
        x1 = max(0.0, (-b) / slope)
        y2 = min(slope * size_of_pic[1] + b, size_of_pic[0])
    if slope < 0:
        x1 = min(size_of_pic[1], (-b) / slope)
        y2 = min(b, size_of_pic[0])

    y1 = slope * x1 + b
    x2 = (y2 - b) / slope
    return [x1, y1, x2, y2, line[4]]


"""
input: image_lines as a matrix, dimension specifying the background size (same size as the original image highly 
       recommended)
output: resulting binary image after drawing as a matrix

"""


def draw_lines(image_lines, dimension):
    result = np.zeros(dimension)
    for x1, y1, x2, y2, _ in image_lines:
        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)
        cv2.line(result, (x1, y1), (x2, y2), 255, 2)
    return result


"""

Find intersection point of two lines
input: line1, line2 as two arrays representing two lines correspondingly, with the format of [x1,y1,x2,y2,angle]
output: x,y as the coordinate of the intersection point of line1 and line2, None if there is no intersection or if two lines overlap

"""


def intersection_shapely(line1, line2):
    l1 = LineString([(line1[0], line1[1]), (line1[2], line1[3])])
    l2 = LineString([(line2[0], line2[1]), (line2[2], line2[3])])
    int_pt = l1.intersection(l2)
    if int_pt.is_empty:
        return None

    point_of_intersection = int_pt.x, int_pt.y
    return point_of_intersection


"""

Find intersections between multiple lines 
input: lines as a matrix
output: a list of coordinates, representing all the intersection points

"""


def find_intersections(lines):
    intersections = []
    for i, line_1 in enumerate(lines):
        for line_2 in lines[i + 1:]:
            if line_1 != line_2:
                intersection = intersection_shapely(line_1, line_2)
                if intersection:  # If lines cross, then add
                    intersections.append(intersection)

    return intersections


"""

Given the image and intersections, find the grid where most intersections occur and treat as vanishing point
input: img as an image, grid_size as the side length of the grid, intersections as an array of all the intersection points in the image
output: the center coordinate of the best cell, the image of the scene with the best cell highlighted

"""


def find_vanishing_point(img, grid_size, intersections, show_step):
    # Image dimensions
    image_height = img.shape[0]
    image_width = img.shape[1]

    # Grid dimensions
    grid_rows = image_height // grid_size
    grid_columns = image_width // grid_size

    # Current cell with most intersection points
    max_intersections = 0
    best_cell = (0.0, 0.0)

    for i, j in itertools.product(range(grid_rows), range(grid_columns)):
        cell_left = j * grid_size
        cell_right = (j + 1) * grid_size
        cell_bottom = i * grid_size
        cell_top = (i + 1) * grid_size
        cv2.rectangle(img, (cell_left, cell_bottom), (cell_right, cell_top),0, 10)

        current_intersections = 0  # Number of intersections in the current cell
        for x, y in intersections:
            if cell_left < x < cell_right and cell_bottom < y < cell_top:
                current_intersections += 1

        # Current cell has more intersections that previous cell (better)
        if current_intersections > max_intersections:
            max_intersections = current_intersections
            best_cell = ((cell_left + cell_right) / 2, (cell_bottom + cell_top) / 2)
            if show_step:
                print("Best Cell:", best_cell)

    if best_cell[0] is not None and best_cell[1] is not None:
        rx1 = int(best_cell[0] - grid_size / 2)
        ry1 = int(best_cell[1] - grid_size / 2)
        rx2 = int(best_cell[0] + grid_size / 2)
        ry2 = int(best_cell[1] + grid_size / 2)
        cv2.rectangle(img, (rx1, ry1), (rx2, ry2), 255, 10)

    return best_cell, img, max_intersections


def extract_features(image, show_step=False):
    height, width = image.shape
    edges = detect_edge(image)
    hash_value = imagehash.average_hash(Image.fromarray(image))
    lines = detect_lines(edges)
    extended = extend_lines(lines, image.shape)
    intersections = find_intersections(extended)
    cell, img, intersections = find_vanishing_point(image, 50, intersections,show_step)

    info = {
        "shape": image.shape,
        "height": height,
        "width": width,
        "lines": lines,
        "vanishing_point": cell,
        "hash": hash_value
    }

    if show_step:
        show_img(edges, "Edges")
        cv2.imwrite("Edge.jpg", edges)
        show_img(draw_lines(lines, image.shape), "Original Lines")
        cv2.imwrite("Lines.jpg", draw_lines(lines, image.shape))
        show_img(draw_lines(extended, image.shape), "Extended Lines")
        cv2.imwrite("Extended Lines.jpg", draw_lines(extended,image.shape))
        show_img(img, "Vanishing Point")
        cv2.imwrite("Vanishing Point.jpg", img)

    return info
