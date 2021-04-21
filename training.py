import numpy as np
import cv2
import os
import NeuralNetwork
import values
import processing as p

X_denom_new = []
Y_denom_new = []
X_denom_old = []
Y_denom_old = []
old = False
img_dir = "training"

for filename in os.listdir(img_dir):
    print("\n" + filename)
    img = cv2.imread(img_dir + "/" + filename)
    # only train with images in the correct rotation
    if filename[5] != 'R':
        # Pre-process image
        _, img_blur = p.preprocess(img)
        y_denomination = [0, 0, 0, 0, 0]
        value = filename[:3]
        if value == '010':
            y_denomination[0] = 1
        elif value == '020':
            y_denomination[1] = 1
        elif value == '050':
            y_denomination[2] = 1
        elif value == '100':
            y_denomination[3] = 1
        elif value == '200':
            y_denomination[4] = 1
        # New/Old note
        if filename[4] == "N":
            Y_denom_new.append(y_denomination)
            old = False
        else:
            Y_denom_old.append(y_denomination)
            old = True
        # Get correct feature masks
        features = []
        if filename[3] == "F":
            if old:
                features = values.features_front_old
            else:
                features = values.features_front_new
        else:
            if old:
                features = values.features_back_old
            else:
                features = values.features_back_new
        # Get normalized mean colours per channel
        mean_colours = np.array(cv2.mean(img)) / 255
        # Get features from feature masks
        if old:
            x_denom_old = p.get_normalized_features(img_blur, features)
            x_denom_old = np.append(x_denom_old, mean_colours[0])
            x_denom_old = np.append(x_denom_old, mean_colours[1])
            x_denom_old = np.append(x_denom_old, mean_colours[2])
            X_denom_old.append(x_denom_old)
        else:
            x_denom_new = p.get_normalized_features(img_blur, features)
            x_denom_new = np.append(x_denom_new, mean_colours[0])
            x_denom_new = np.append(x_denom_new, mean_colours[1])
            x_denom_new = np.append(x_denom_new, mean_colours[2])
            X_denom_new.append(x_denom_new)

iterations = 10000

nn_new = NeuralNetwork.NeuralNetwork(np.array(X_denom_new), np.array(Y_denom_new))
for i in range(iterations):
    nn_new.feedforward()
    nn_new.backpropagation()

# np.set_printoptions(precision=5, suppress=True)
# print(nn_new.output)
np.savetxt('nn_new_weights1.csv', nn_new.weights1, delimiter=',')
np.savetxt('nn_new_weights2.csv', nn_new.weights2, delimiter=',')

nn_old = NeuralNetwork.NeuralNetwork(np.array(X_denom_old), np.array(Y_denom_old))
for i in range(iterations):
    nn_old.feedforward()
    nn_old.backpropagation()

np.savetxt('nn_old_weights1.csv', nn_old.weights1, delimiter=',')
np.savetxt('nn_old_weights2.csv', nn_old.weights2, delimiter=',')

print("complete")
