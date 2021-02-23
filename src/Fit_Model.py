#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 04:28:09 2020

@author: jiuqiwang
"""

from src.Process_Img import load_img, extract_features
import math

'''
a function that loads all the models and extract the features
Output: a list of feature dictionaries
'''


def load_model():
    models = load_img(folder="../model/")
    features = [extract_features(model) for model in models]
    return features


'''
Input: lines as a numpy array, width as an integer, height as an integer
Output: four lists containing lines in four quadrants
'''


def divide_quadrant(lines, width, height):
    first = []  # first quadrant line
    second = []  # second quadrant line
    third = []  # third quadrant line
    fourth = []  # fourth quadrant line
    for line in lines:
        x1, y1, x2, y2, _ = line
        if (x1 <= width / 2 and y1 <= height / 2) or (x2 <= width / 2 and y2 <= height / 2):
            first.append(line)
        elif (x1 >= width / 2 and y1 <= height / 2) or (x2 >= width / 2 and y2 <= height / 2):
            second.append(line)
        elif (x1 >= width / 2 and y1 >= height / 2) or (x2 >= width / 2 and y2 >= height / 2):
            third.append(line)
        else:
            fourth.append(line)

    return first, second, third, fourth


'''
Input: input_line as a (1,5) numpy array, line_set as an (n,5) numpy array
Output: the line that has the smallest angular difference with the input line
'''


def closest_angle_line(input_line, line_set):
    if not line_set:
        return None

    closest = None
    min_angle_diff = 360
    index = -1
    for i, some_line in enumerate(line_set):
        angle_diff = abs((some_line[4] % 180) - input_line[4] % 180)
        if angle_diff < min_angle_diff:
            closest = some_line
            min_angle_diff = angle_diff
            index = i

    del line_set[index]
    return closest


'''
Input: line as a (1,5) numpy array
Output: the length of the line
'''


def length(line):
    x1, y1, x2, y2, _ = line
    line_length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return line_length


'''
Input: the input lines and model lines as (n,5) numpy arrays, input_shape and model_shape as tuples
Output: the cost of lines
'''


def line_cost(input_lines, model_lines, input_shape, model_shape):
    input_height, input_width = input_shape
    model_height, model_width = model_shape
    input_quadrants = divide_quadrant(input_lines, input_width, input_height)
    model_quadrants = divide_quadrant(model_lines, model_width, model_height)
    # get diagonal length
    input_diagonal_length = math.sqrt(input_height ** 2 + input_width ** 2)
    model_diagonal_length = math.sqrt(model_height ** 2 + model_width ** 2)

    cost = 0
    for i in range(4):
        input_quadrant = input_quadrants[i]
        model_quadrant = model_quadrants[i]
        for line in input_quadrant:
            closest = closest_angle_line(line, model_quadrant)
            if closest is not None:
                input_length_ratio = length(line) / input_diagonal_length
                model_length_ratio = length(closest) / model_diagonal_length
                cost += abs(input_length_ratio - model_length_ratio) + abs((closest[4] % 180) - (line[4] % 180)) / 360
            else:
                cost += length(line) / input_diagonal_length
        # add the cost
        for line in model_quadrant:
            cost += length(line) / model_diagonal_length

    return cost


'''
Input: input_vanishing, model_vanishing as tuples representing coordinates, input_shape, model_shape as tuples 
       representing dimensions
Output: the vanishing point cost
'''


def vanishing_point_cost(input_vanishing, model_vanishing, input_shape, model_shape):
    input_x, input_y = input_vanishing
    model_x, model_y = model_vanishing
    input_height, input_width = input_shape
    model_height, model_width = model_shape
    # compute the absolute difference
    x_cost = abs((input_x / input_width) - (model_x / model_width))
    y_cost = abs((input_y / input_height) - (model_y / model_height))
    return x_cost + y_cost


'''
Input: input_hash and model_hash as 64-bit integers
Output: hash cost
'''


def hash_cost(input_hash, model_hash):
    return abs(input_hash - model_hash)


'''
Input: input features and model features as dictionaries
Output: the total cost
'''


def total_cost(input_features, model_features):
    cost_of_line = line_cost(input_features['lines'], model_features['lines'],
                             input_features['shape'], model_features['shape'])

    cost_of_vanishing = vanishing_point_cost(input_features['vanishing_point'], model_features['vanishing_point'],
                                             input_features['shape'], model_features['shape'])

    cost_of_hash = hash_cost(input_features['hash'], model_features['hash'])

    return cost_of_line + 10 * cost_of_hash + cost_of_vanishing


'''
Input: image object, threshold
Output: a boolean indicating whether the image belongs to the class
'''


def predict(image, threshold=250, show_step=False):
    models = load_model()
    costs = []
    image_feature = extract_features(image, show_step)
    for model in models:
        costs.append(total_cost(image_feature, model))
    return min(costs) < threshold


if __name__ == '__main__':
    img = load_img("Hallway2.1")
    predict(img[0], show_step=True)
