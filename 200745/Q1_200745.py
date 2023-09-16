import cv2
import numpy as np

# def show_vertices(img, top_left, top_right, bottom_left, bottom_right):
#     img = cv2.circle(img, top_left, 5, (0,0,255), -1)
#     img = cv2.circle(img, top_right, 5, (0, 255, 0), 2)
#     img = cv2.circle(img, bottom_left, 5, (255, 0, 0), -1)
#     img = cv2.circle(img, bottom_right, 5, (255, 255, 255), 2)
#     show_image(img)

def dist(p1, p2):
    return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def dist_color(c1, c2):
    return ((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2 + (c1[2]-c2[2])**2)**0.5

def clean_image(image):
    for i in range(0, image.shape[0]):
        for j in range(0, image.shape[1]):
            orange = dist_color(image[i][j], [51, 153, 255])
            white = dist_color(image[i][j], [255, 255, 255])
            green = dist_color(image[i][j], [0, 128, 0])
            blue = dist_color(image[i][j], [255, 0, 0])
            black = dist_color(image[i][j], [0, 0, 0])

            min_dist = min(orange, white, green, blue, black)
            if min_dist == orange:
                image[i][j] = [51, 153, 255]
            elif min_dist == white:
                image[i][j] = [255, 255, 255]
            elif min_dist == green:
                image[i][j] = [0, 128, 0]
            elif min_dist == blue:
                image[i][j] = [255, 0, 0]
            elif min_dist == black:
                image[i][j] = image[i+3][j+3]      
    return image

def show_image(image):
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # print("Done")

def solution(image_path):
    image= cv2.imread(image_path)
    # print(image)
    ######################################################################
    ######################################################################
    #####  WRITE YOUR CODE BELOW THIS LINE ###############################

    width = image.shape[1]
    height = image.shape[0] 

    m = 25 #Can be increased if edges smoother

    top_left = (width, m*height)
    top_right = (0,(height + width)*m)
    bottom_left =  (width + height, 0)
    bottom_right = (0,0)

    bg = [0,0,0]
    for i in range(0, height):
        for j in range(0, width):
            if((image[i][j] != bg).any()):
                summ = m*i+j
                diff = m*i-j
                if(summ < (top_left[0]+ m*top_left[1])):
                    top_left = (j,i)
                if(diff < (m*top_right[1]-top_right[0])):
                    top_right = (j,i)
                if(diff > (m*bottom_left[1]-bottom_left[0])):
                    bottom_left = (j,i)
                if(summ > (bottom_right[0]+ m*bottom_right[1])):    
                    bottom_right = (j,i)

    if dist(top_left, top_right)<36 or dist(bottom_right, bottom_left)<36 or dist(top_left, bottom_right)<36 or dist(top_right,bottom_left)<36 or dist(top_left,bottom_left)<36 or dist(top_right, bottom_right)<36: #if both are within 6 pixel of each other (in a respectable size image), then change slope as image is horizontal
        m = 1/25

        top_left = (width, m*height)
        top_right = (0,(height + width)*m)
        bottom_left =  (width + height, 0)
        bottom_right = (0,0)

        bg = [0,0,0]
        for i in range(0, height):
            for j in range(0, width):
                if((image[i][j] != bg).any()):
                    summ = m*i+j
                    diff = m*i-j
                    if(summ < (top_left[0]+ m*top_left[1])):
                        top_left = (j,i)
                    if(diff < (m*top_right[1]-top_right[0])):
                        top_right = (j,i)
                    if(diff > (m*bottom_left[1]-bottom_left[0])):
                        bottom_left = (j,i)
                    if(summ > (bottom_right[0]+ m*bottom_right[1])):    
                        bottom_right = (j,i)
    # print(top_left, top_right, bottom_left, bottom_right) 
    # show_vertices(image, top_left, top_right, bottom_left, bottom_right)

    # Perspective Transform using opencv and vertices calculated above
    pts1 = np.float32([top_left, top_right, bottom_left, bottom_right])
    pts2 = np.float32([[0,0],[width,0],[0,height],[width,height]])
    matrix = cv2.getPerspectiveTransform(pts1,pts2)
    image = cv2.warpPerspective(image,matrix,(width,height))
    # show_image(image)

    image = cv2.resize(image, (600,600))

    ######################################################################
    # image = clean_image(image)
    return image


# ## Run function
show_image(solution("Q1/test/1.png"))
# show_image(solution("Q1/test/2.png"))
# show_image(solution("Q1/test/3.png"))
# show_image(solution("Q1/test/4.png"))