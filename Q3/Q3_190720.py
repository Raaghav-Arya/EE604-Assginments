import cv2
import numpy as np
import os

def show_image(image):
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("Done")


def solution(image_path):
    ############################
    ############################
    
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # grayscale mei convert karo
    gray = cv2.bitwise_not(gray) # grayscale image invert karo taki thresholding sahi se ho
    thresh_val, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) # perform OTSU thresholding

    ## Testing code to display the image
    # print("Threshold value: ", thresh_val)

    show_image(thresh)

    
    ############################
    ############################
    ## comment the line below before submitting else your code wont be executed##
    # pass
    return image

# for filename in os.listdir('Q3/test'):
#     image_path = os.path.join('Q3/test', filename)
#     solution(image_path)

solution('Q3/test/3_a.png')