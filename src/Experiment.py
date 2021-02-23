from src.Fit_Model import predict, load_img, load_model
import numpy as np

'''
load models
'''

models = load_model()

'''
load validation set and test set
'''
validation_positive = load_img(folder='../validation-positive/')
validation_negative = load_img(scene_name="*", folder='../validation-negative/')
test_positive = load_img(folder='../test-positive/')
test_negative = load_img(scene_name="*", folder='../test-negative/')

'''
run the threshold localization algorithm
Output: a list containing tuples of (threshold, accuracy)
'''


def experiment_threshold():
    thresholds = np.linspace(250, 300, 6)
    all_accuracy = []
    for threshold in thresholds:
        true_positive = 0
        true_negative = 0

        for img in validation_positive:
            if predict(img, threshold):
                true_positive += 1

        for img in validation_negative:
            if not predict(img, threshold):
                true_negative += 1

        total = len(validation_positive) + len(validation_negative)
        accuracy = (true_positive + true_negative) / total
        all_accuracy.append((threshold, accuracy))

    return all_accuracy


'''
run the final test
Output: accuracy
'''


def run_test():
    true_positive = 0
    true_negative = 0
    for img in test_positive:
        if predict(img, 250):
            true_positive += 1

    for img in test_negative:
        if not predict(img, 250):
            true_negative += 1

    print("True Positive: " + str(true_positive))
    print("True Negative: " + str(true_negative))

    total = len(test_positive) + len(test_negative)
    return (true_positive + true_negative) / total


if __name__ == '__main__':
    print(run_test())
