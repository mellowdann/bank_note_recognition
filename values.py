import numpy as np

# Features for determining Front/Back Old/New and Rotation
region1 = [0, 99, 50, 99]
region2 = [300, 399, 50, 99]
region3 = [200, 299, 0, 199]
region4 = [300, 399, 100, 149]
region5 = [0, 99, 100, 149]
regions = [region1, region2, region3, region4, region5]

# New Front feature mask regions of interest
nf_coat_of_arms = [9, 53, 7, 55]
nf_mandela = [198, 303, 32, 164]
nf_bank_text = [44, 283, 14, 30]
nf_value_left = [64, 139, 127, 179]
nf_value_right = [309, 377, 127, 176]
nf_lines_left = [0, 20, 123, 174]
nf_lines_right = [379, 499, 127, 172]
nf_value_words = [359, 379, 14, 124]
nf_pattern_left = [111, 160, 33, 73]
nf_pattern_right = [328, 359, 15, 124]
nf_pattern_bottom = [96, 369, 179, 199]
nf_animal_half = [21, 78, 143, 176]
features_front_new = [nf_coat_of_arms, nf_mandela, nf_bank_text,
                      nf_value_left, nf_value_right, nf_lines_left, nf_lines_right, nf_value_words,
                      nf_pattern_left, nf_pattern_right, nf_pattern_bottom, nf_animal_half]

# New Back feature mask regions of interest
nb_animal = [95, 204, 37, 171]
nb_value_left = [35, 71, 9, 35]
nb_value_right = [205, 276, 121, 174]
nb_value_text = [376, 399, 11, 102]
nb_half_animal = [330, 374, 140, 175]
nb_animal_pattern = [282, 329, 123, 184]
nb_text_top = [65, 259, 13, 31]
nb_text_bottom = [32, 258, 169, 186]
nb_pattern_left = [35, 92, 46, 173]
nb_pattern_right = [209, 287, 41, 86]
nb_pattern_top = [35, 204, 0, 13]
nb_pattern_bottom = [35, 204, 187, 199]
features_back_new = [nb_animal, nb_half_animal, nb_animal_pattern,
                     nb_value_left, nb_value_right, nb_value_text, nb_text_top, nb_text_bottom,
                     nb_pattern_left, nb_pattern_right, nb_pattern_top, nb_pattern_bottom]

# Old Front feature mask regions of interest
of_animal = [235, 335, 19, 138]
of_small_animal = [108, 190, 35, 111]
of_value_left = [105, 190, 115, 173]
of_value_text_left = [0, 24, 3, 114]
of_value_text_right = [374, 399, 3, 117]
of_circle_left = [7, 69, 115, 171]
of_circle_right = [329, 390, 117, 170]
of_dots = [2, 59, 175, 193]
features_front_old = [of_animal, of_small_animal,
                      of_value_left, of_value_text_left, of_value_text_right,
                      of_circle_left, of_circle_right, of_dots]

# Old Back feature mask regions of interest
ob_value_text_left = [0, 21, 0, 120]
ob_value_text_right = [363, 399, 0, 120]
ob_value_left = [0, 22, 166, 199]
ob_value_right = [371, 399, 167, 199]
ob_value_mid = [186, 299, 111, 182]
ob_pattern_large = [52, 287, 29, 114]
ob_pattern_small = [27, 185, 114, 182]
ob_circle = [331, 392, 116, 176]
features_back_old = [ob_value_text_left, ob_value_text_right, ob_value_left, ob_value_right, ob_value_mid,
                     ob_pattern_large, ob_pattern_small, ob_circle]

# Colours used to classify notes
colours = np.array([[175, 202, 175], [105, 139, 73], [215, 234, 220], [120, 162, 118],
                    [171, 199, 168], [108, 140, 66], [219, 239, 227], [127, 156, 136],
                    [154, 186, 210], [76, 114, 163], [211, 235, 245], [124, 157, 185],
                    [141, 178, 207], [78, 113, 160], [194, 226, 238], [107, 135, 160],
                    [173, 182, 216], [91, 91, 186], [222, 227, 240], [144, 134, 192],
                    [169, 179, 216], [94, 89, 174], [219, 229, 243], [129, 137, 187],
                    [201, 188, 179], [141, 102, 79], [231, 225, 220], [153, 126, 146],
                    [203, 188, 175], [138, 100, 82], [224, 230, 227], [152, 132, 145],
                    [171, 185, 215], [76, 113, 202], [207, 234, 241], [121, 152, 201],
                    [161, 185, 220], [73, 108, 196], [193, 232, 245], [104, 147, 195]])
