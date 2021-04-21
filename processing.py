import cv2
import numpy as np
import values
from skimage.feature import greycomatrix, greycoprops
import sklearn.preprocessing as sk


def get_note_category_and_correct_rotation(src):
    image = src.copy()
    region1 = image[values.region1[2]:values.region1[3], values.region1[0]:values.region1[1]]
    region2 = image[values.region2[2]:values.region2[3], values.region2[0]:values.region2[1]]
    white1 = np.sum(region1) / 255
    white2 = np.sum(region2) / 255
    if white1 < white2:
        image = cv2.rotate(image, cv2.ROTATE_180)
        flipped = True
    else:
        flipped = False

    # New or old note
    region3 = image[values.region3[2]:values.region3[3], values.region3[0]:values.region3[1]]
    white3 = np.sum(region3) / 255
    # True: New note
    # False: Old note
    if white3 > 300:
        old = False
        # Front or back of new note
        region4 = image[values.region4[2]:values.region4[3], values.region4[0]:values.region4[1]]
        region2 = image[values.region2[2]:values.region2[3], values.region2[0]:values.region2[1]]
        white4 = np.sum(region4) / 255
        white2 = np.sum(region2) / 255
        # True: Front, no rotation needed
        # False: Back, rotation needed
        if white4 > white2:
            front = True
            need_rotation = flipped
        else:
            front = False
            need_rotation = not flipped
    else:
        old = True
        # Front or back of old note
        region1 = image[values.region1[2]:values.region1[3], values.region1[0]:values.region1[1]]
        region5 = image[values.region5[2]:values.region5[3], values.region5[0]:values.region5[1]]
        white1 = np.sum(region1) / 255
        white5 = np.sum(region5) / 255
        # True: Front, no rotation needed
        # False: Back, rotation needed
        if white1 > white5:
            front = True
            need_rotation = flipped
        else:
            front = False
            need_rotation = not flipped
    return old, front, need_rotation


def get_features_old(img_canny, features):
    x = []
    for feature in features:
        ROI = img_canny[feature[2]:feature[3], feature[0]:feature[1]]
        height, width = ROI.shape
        countRows = 0
        countColumns = 0
        for i in range(height):
            previousColumn = 0
            for j in range(width):
                k = ROI[i, j]
                if k != previousColumn:
                    previousColumn = k
                    countColumns += 1
        for j in range(width):
            previousRow = 0
            for i in range(height):
                k = ROI[i, j]
                if k != previousRow:
                    previousRow = k
                    countRows += 1
        # normalize features to range 0 - 1
        max_size = width * height
        columns = countColumns / max_size
        rows = countRows / max_size
        # add features to ML input vector
        x.append(columns)
        x.append(rows)
    return x


def get_normalized_features(img_gray_equ, features):
    x = []
    for feature in features:
        ROI = img_gray_equ[feature[2]:feature[3], feature[0]:feature[1]]
        glcm = greycomatrix(ROI, distances=[1], angles=[0], levels=256)
        x.append(greycoprops(glcm, 'contrast')[0][0])
        x.append(greycoprops(glcm, 'dissimilarity')[0][0])
        x.append(greycoprops(glcm, 'homogeneity')[0][0])
        x.append(greycoprops(glcm, 'energy')[0][0])
        x.append(greycoprops(glcm, 'correlation')[0][0])
    x = np.array(x)
    oldshape = x.shape
    x = sk.normalize(np.reshape(x, (1, -1)), axis=1)
    x = np.reshape(x, oldshape)
    # print('\n')
    # print("max: {:.9f}".format(np.max(x)))
    # print("max: {:.9f}".format(np.min(x)))
    return x


def preprocess(img):
    img_scaled = cv2.resize(img, (400, 200), interpolation=cv2.INTER_AREA)
    img_gray = cv2.cvtColor(img_scaled, cv2.COLOR_BGR2GRAY)
    img_equalized = cv2.equalizeHist(img_gray)
    _, img_binary = cv2.threshold(img_equalized, 180, 255, cv2.THRESH_BINARY)
    img_blur = cv2.GaussianBlur(img_equalized, (7, 7), 0)
    return img_binary, img_blur


def get_classification_from_colour(img, colours):
    Z = np.float32(img.reshape((-1, 3)))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 2
    _, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))
    colour1 = np.array(res2[0][0])
    colour2 = np.array(res2[0][0])
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if any(res2[i][j] != colour1):
                colour2 = np.array(res2[i][j])
                break
    # Calculate two closest colours
    distances1 = get_normalized_euclidean_distance(colours, colour1)
    distances2 = get_normalized_euclidean_distance(colours, colour2)
    # combine into single value for each note option
    temp = np.zeros(20)
    for i in range(0, 40, 2):
        temp[int(i / 2)] = max(distances1[i], distances1[i + 1]) * max(distances2[i], distances2[i + 1])
    # choose the best note of each denomination
    percentages = np.zeros(5)
    for i in range(0, 20, 4):
        percentages[int(i / 4)] = max(temp[i], temp[i + 1], temp[i + 2], temp[i + 3])

    return percentages


def get_normalized_euclidean_distance(point1, point2):
    distances = np.sqrt(np.sum((point1 - point2) ** 2, axis=1))
    return np.clip((255 - distances) / 255, 0, 1)


def get_classification_from_filename(filename):
    classification = np.array(["fake", "-", "-", "-"], dtype='U6')
    if filename[:4] != 'fake':
        classification[0] = filename[:3]
        classification[1] = "Front" if filename[3] == "F" else "Back"
        classification[2] = "New" if filename[4] == "N" else "Old"
        classification[3] = "Rotate" if filename[5] == "R" else "-"
    return classification
