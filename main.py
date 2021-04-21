import numpy as np
import cv2
import os
import NeuralNetwork
import values
import processing as p
import time

# Authors:
# Shaneer Luchman - 216003502
# Daniel Mundell - 220108104
# Siphokazi H. Nene - 216009670
# Akaylan Perumal - 216035695

# Change these variables to determine what is printed
print_correct = False
print_incorrect = True

_denom = ["010", "020", "050", "100", "200"]
_type = ["New", "Old"]
_side = ["Front", "Back"]

x = np.zeros((1, 63), dtype=int)
y = np.array([[0, 0, 0, 0, 0]])
nn_new = NeuralNetwork.NeuralNetwork(x, y)
nn_new.weights1 = np.genfromtxt('nn_new_weights1.csv', delimiter=',')
nn_new.weights2 = np.genfromtxt('nn_new_weights2.csv', delimiter=',')

x = np.zeros((1, 43), dtype=int)
nn_old = NeuralNetwork.NeuralNetwork(x, y)
nn_old.weights1 = np.genfromtxt('nn_old_weights1.csv', delimiter=',')
nn_old.weights2 = np.genfromtxt('nn_old_weights2.csv', delimiter=',')

count_total = 0
count_correct = 0

# Calculate time for performance
time_start = time.process_time()

img_dir = "RandNotes"
for filename in os.listdir(img_dir):
    img = cv2.imread(img_dir + "/" + filename)
    # Pre-process and enhancement
    img_binary, img_blur = p.preprocess(img)
    # Use Decision Tree Classifier
    # new/old, front/back, rotate/no rotate
    old, front, need_rotation = p.get_note_category_and_correct_rotation(img_binary)
    if need_rotation:
        img_blur = cv2.rotate(img_blur, cv2.ROTATE_180)
    prediction = np.array(["", "", "", ""], dtype='U6')
    prediction[3] = "Rotate" if need_rotation else "-"
    # Choose and use the correct neural network as a first classification
    if not old:
        prediction[2] = "New"
        if front:
            prediction[1] = "Front"
            features = values.features_front_new
        else:
            prediction[1] = "Back"
            features = values.features_back_new
    else:
        prediction[2] = "Old"
        if front:
            prediction[1] = "Front"
            features = values.features_front_old
        else:
            prediction[1] = "Back"
            features = values.features_back_old
    x = p.get_normalized_features(img_blur, features)
    # Get normalized mean colours per channel
    mean_colours = np.array(cv2.mean(img)) / 255
    x = np.append(x, mean_colours[0])
    x = np.append(x, mean_colours[1])
    x = np.append(x, mean_colours[2])
    if not old:
        denom_prediction = nn_new.predict(x)
    else:
        denom_prediction = nn_old.predict(x)

    # Use k-nearest neighbours as a second classification
    colour_percentages = p.get_classification_from_colour(img, values.colours)

    # Get predictions from NN and K-NN
    prediction[0] = _denom[np.argmax(denom_prediction)]
    colour_prediction = _denom[np.argmax(colour_percentages)]

    # Compare predictions and NN percentage to determine genuine or non-genuine
    genuine_prediction = prediction[0] == colour_prediction
    if denom_prediction[np.argmax(denom_prediction)] < 0.7:
        genuine_prediction = False

    # Get the real classification and check if our prediction is correct
    classification = p.get_classification_from_filename(filename)
    genuine_note = filename[0:4] != 'fake'
    genuine_result = genuine_prediction == genuine_note
    result = prediction == classification
    colour_result = colour_prediction == classification[0]
    overall_result = result[0] == colour_result and result[0]
    overall_prediction = colour_prediction
    if not overall_result:
        overall_prediction = "-"

    if genuine_result:
        correct = all(result) or not genuine_note
    else:
        correct = False

    # Print results
    if (print_correct and correct) or (print_incorrect and not correct):
        print("\n")
        print('Image {}: {}'.format(count_total, filename))
        print('{:14} {:7} {:10} {:7}'.format("", "Actual", "Predicted", "Result"))
        print(
            '{:14} {:7} {:10} {:7}'.format("Genuine:", str(genuine_note), str(genuine_prediction), str(genuine_result)))
        print(
            '{:14} {:7} {:10} {:7}'.format("Denomination:", classification[0], overall_prediction, str(overall_result)))
        print('{:14} {:7} {:10} {}'.format("Colour:", classification[0], colour_prediction, colour_result))
        print('{:14} {:7} {:10} {} {:5f}'.format("GLCM:", classification[0], prediction[0], result[0],
                                                 denom_prediction[np.argmax(denom_prediction)]))
        if filename[0:4] != 'fake':
            print('{:14} {:7} {:10} {}'.format("Front/Back:", classification[1], prediction[1], result[1]))
            print('{:14} {:7} {:10} {}'.format("Old/New:", classification[2], prediction[2], result[2]))
            print('{:14} {:7} {:10} {}'.format("Rotate:", classification[3], prediction[3], result[3]))

    # Keep track of overall stats
    count_total += 1
    if correct:
        count_correct += 1

# Print overall stats
print('\n')
print('Total: {}'.format(count_total))
print('Correct: {}'.format(count_correct))
print('Incorrect: {}'.format(count_total - count_correct))

time_end = time.process_time()
duration = time_end - time_start
# Print elapsed time
print("\nTotal elapsed time: {}".format(duration))
print("Average elapsed time: {}".format(duration / count_total))
