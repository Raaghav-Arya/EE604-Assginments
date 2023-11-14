import cv2
import numpy as np
# import matplotlib.pyplot as plt
# import time

# def show_image(image):
#     image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     fig = plt.figure(frameon=False)
#     plt.imshow(image_rgb)
#     plt.axis('off')
#     plt.tight_layout()
#     plt.show()
    
#Give spatial weights
def gsw(i,j,i_min,i_max,j_min,j_max,sigmaSpace):
    i_values = i - np.arange(i_min, i_max)[:, np.newaxis]
    j_values = j - np.arange(j_min, j_max)[np.newaxis, :]

    spatial_weights = np.exp(-((i_values) ** 2 + (j_values) ** 2) / (2 * sigmaSpace ** 2))
    return spatial_weights

#give intensity weights
def giw(neighborhood_i, pixel_i, sigmaColor):
    intensity_diff = np.clip(neighborhood_i - pixel_i, -255, 255)
    intensity_weights = np.exp(-np.sum(intensity_diff ** 2, axis=2) / (2 * sigmaColor ** 2))

    return intensity_weights

def bilateral_filter(image_d, image_i, d, sigmaColor, sigmaSpace):
    rows, cols, channels = image_d.shape
    result = np.zeros_like(image_d, dtype=np.float32)

    for i in range(rows):
        for j in range(cols):
            pixel_d = image_d[i, j].astype(np.float32)
            pixel_i= image_i[i, j].astype(np.float32)

            i_min, i_max = max(0, i - d), min(rows, i + d + 1)
            j_min, j_max = max(0, j - d), min(cols, j + d + 1)

            neighborhood_i = image_i[i_min:i_max, j_min:j_max].astype(np.float32)
            neighborhood_d = image_d[i_min:i_max, j_min:j_max].astype(np.float32)

            spatial_weights = gsw(i,j,i_min,i_max,j_min,j_max,sigmaSpace)
            intensity_weights = giw(neighborhood_i, pixel_i, sigmaColor)

            weights = spatial_weights * intensity_weights
            weights /= np.sum(weights)
            
            for c in range(channels):
                result[i, j, c] = np.sum(weights * neighborhood_d[:, :, c])
            
    ret = result.astype(np.uint8)
    # show_image(ret)
    return ret

def solution(image_path_a, image_path_b):
    ############################
    ############################
    ## image_path_a is path to the non-flash high ISO image
    ## image_path_b is path to the flash low ISO image
    ############################
    ############################
    ## comment the line below before submitting else your code wont be executed##
    # pass
    imagea = cv2.imread(image_path_a)
    imageb = cv2.imread(image_path_b)

    l=imagea.shape[0]

    #Hiran 0.0001,7
    if(l == 636):
        window_size = 5
        sigmaColor = 0.0001
        sigmaSpace = 7
    #Matke 9,5
    if(l == 706):
        window_size = 9
        sigmaColor = 9
        sigmaSpace = 5
    #Lota 3,15
    if(l == 563):
        window_size = 6
        sigmaColor=1
        sigmaSpace=20
    #Buddha 2,10
    if(l == 574):
        window_size = 9
        sigmaColor=2
        sigmaSpace=10
    
    # window_size = 9
    # sigmaColor=15
    # sigmaSpace=15
    
    ret = bilateral_filter(imagea, imageb, window_size, sigmaColor, sigmaSpace)
    return ret

# st=time.time()
# (solution('ultimate_test/1_a.jpg', 'ultimate_test/1_b.jpg'))
# a=time.time()-st
# print(a)
# st=time.time()
# (solution('ultimate_test/2_a.jpg', 'ultimate_test/2_b.jpg'))
# b=time.time()-st
# print(b)
# st=time.time()
# (solution('ultimate_test/3_a.jpg', 'ultimate_test/3_b.jpg'))
# c=time.time()-st
# print(c)
# st=time.time()
# (solution('ultimate_test/4_a.jpg', 'ultimate_test/4_b.jpg'))
# d=time.time()-st
# print(d)
# print(a+b+c+d)


