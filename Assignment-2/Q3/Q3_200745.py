import cv2
import numpy as np
# import matplotlib.pyplot as plt

# def show_image(image):
#     image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     fig = plt.figure(frameon=False)
#     plt.imshow(image_rgb)
#     plt.axis('off')
#     plt.tight_layout()
#     plt.show()
    

def Closing(mask):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    closed_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return closed_mask

def bding_rect(img,thresh):
    # Rectangle on biggest ravan (Matlab alag se sar hoga to usse include nhi karega)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x1, y1, w1, h1 = cv2.boundingRect(contour)
    # cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Rectangle including all ravan heads
    merged_contours = np.vstack([contours[i] for i in range(len(contours))])
    x, y, w, h = cv2.boundingRect(merged_contours)
    # cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

    if([x,y,w,h] != [x1,y1,w1,h1]):
        return None
    
    cropped_img = img[y:y+h, x:x+w]
    thresh = thresh[y:y+h, x:x+w]
    # show_image(cropped_img)
    # show_image(thresh)

    return cropped_img,thresh

def non_zero(arr):
    first_non_zero_index = np.argmax(arr != 0)
    last_non_zero_index = len(arr) - 1 - np.argmax(arr[::-1] != 0)

    return first_non_zero_index,last_non_zero_index

def dist_white(arr):
    return ((arr[0]-255)**2 + (arr[1]-255)**2 + (arr[2]-255)**2)**0.5

def solution(image_path):
    ############################
    ############################
    image= cv2.imread(image_path)
    # show_image(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    _, thresh = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
    thresh=Closing(thresh)
    # show_image(thresh)
    
    temp = bding_rect(image.copy(),thresh)
    if(temp == None):
        return 'fake'
    img,thresh = temp  

    # show_image(img)
    # show_image(thresh)

    fir,las = non_zero(thresh[-5,:])
    mid = round(img.shape[1]/((las+fir)//2),3) 
    asp_rat = round(img.shape[1]/img.shape[0],3)
    # print("mid",mid)
    # print("white", dist_white(img[-2,fir+1]))

    rav = "north" if dist_white(img[-5,fir+5]) <200 else "south"

    ideal_asp_rat_south = 2.124
    ideal_asp_rat_north = 1.637

    ideal_mid_south = 2.138
    ideal_mid_north = 2.141

    if rav=="south":
        # print("mid", abs(mid-ideal_mid_south))
        # print("asp", abs(asp_rat-ideal_asp_rat_south))
        if(abs(mid-ideal_mid_south)>0.15 or abs(asp_rat-ideal_asp_rat_south)>0.12):
            return"fake"
    else:
        # print("mid", abs(mid-ideal_mid_north))
        # print("asp", abs(asp_rat-ideal_asp_rat_north))
        if(abs(mid-ideal_mid_north)>0.1 or abs(asp_rat-ideal_asp_rat_north)>0.5):
            return"fake"

        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    # show_image(gray)
    if rav=="north":
        _, thresh = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY)
    else:
        _, thresh = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY)
    
    # show_image(thresh)

    height = thresh.shape[0]
    away_pix = np.sum(thresh[height-height//5:height,:])+np.sum(thresh[0:height//5,:])
    away_pix/=255
    total_pix = np.sum(thresh)/255

    if(away_pix==0):
        return "real"

    face_check = total_pix/away_pix
    # print(face_check)
    if(face_check<100):
        return "fake"

    ############################
    ############################
    ## comment the line below before submitting else your code wont be executed##
    # pass
    return "real"



# print(solution('test/r1.jpg'))

# print(solution('test/r2.jpg'))

# print(solution('test/r3.jpg'))

# print(solution('test/r4.jpg'))

# print(solution('test/r5.jpg'))

# print(solution('test/r6.jpg'))

# print(solution('test/r13.jpg'))

# print(solution('test/r15.jpg'))


