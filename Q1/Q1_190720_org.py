import cv2
import numpy as np

################################################################### Debugging and other attempt
# from skimage.metrics import structural_similarity as ssim

# max(img, key = func) returns the element x from img for which func(x) is maximum 

# def show_vertices(img, top_left, top_right, bottom_left, bottom_right):
#     img = cv2.circle(img, top_left, 5, (0,0,255), -1)
#     img = cv2.circle(img, top_right, 5, (0, 255, 0), 2)
#     img = cv2.circle(img, bottom_left, 5, (255, 0, 0), -1)
#     img = cv2.circle(img, bottom_right, 5, (255, 255, 255), 2)
#     show_image(img)

# def solution_approach1(image_path):
#     image = cv2.imread(image_path)
#     ######################################################################
#     ######################################################################
#     #####  WRITE YOUR CODE BELOW THIS LINE ###############################
    
#     ## Initiation
#     img = cv2.imread(image_path)
#     width = img.shape[1]
#     height = img.shape[0]

#     #Assuming bg is black
#     bg = np.array([0,0,0])  

#     # Debugging
#     # for j in range(0, width):
#     #     l=(img[-1][-(j+1)] != bg).any()
#     #     print(img[-1][-(j+1)], l)    

#     ## flag_vertical checks orientation of image

#     xcentre = width/2
#     ycentre = height/2

#     flag_vertical = False
#     for i in range(0,5):
#         if (img[i][int(xcentre)] != bg).any():
#             flag_vertical = True

#     # print(flag_vertical)

#     ## Approach 1

#     flag1 = False
#     flag2 = False
#     flag3 = False
#     flag4 = False

#     if flag_vertical:
#         for j in range(0,width):
#             for i in range(0,height):
#                 if(i>10): 
#                     break
#                 if((img[i][j] != bg).any() and flag1 == False):
#                     top_left = j,i
#                     flag1 = True
#                 if((img[i][-(j+1)] != bg).any() and flag2 == False):
#                     top_right = (width-j-1),i
#                     flag2 = True
#                 if((img[-(i+1)][j] != bg).any() and flag3 == False):
#                     bottom_left = j,(height-i-1)
#                     flag3 = True
#                 if((img[-(i+1)][-(j+1)] != bg).any() and flag4 == False):
#                     bottom_right = (width-j-1),(height-i-1)
#                     flag4 = True
#                 if(flag1 and flag2 and flag3 and flag4):
#                     break
#             if(flag1 and flag2 and flag3 and flag4):
#                 break
#     else:
#         for i in range(0, height):
#             for j in range(0, width):
#                 if(j>10):
#                     break
#                 if((img[i][j] != bg).any() and flag1 == False):
#                     top_left = j,i
#                     flag1 = True
#                 if((img[i][-(j+1)] != bg).any() and flag2 == False):
#                     top_right = (width-j-1),i
#                     flag2 = True
#                 if((img[-(i+1)][j] != bg).any() and flag3 == False):
#                     bottom_left = j,(height-i-1)
#                     flag3 = True
#                 if((img[-(i+1)][-(j+1)] != bg).any() and flag4 == False):
#                     bottom_right = (width-j-1),(height-i-1)
#                     flag4 = True
#                 if(flag1 and flag2 and flag3 and flag4):
#                     break
#             if(flag1 and flag2 and flag3 and flag4):
#                 break      

#     ## Check if the selected vertice points are correct
#     # show_vertices(img, top_left, top_right, bottom_left, bottom_right)

#     ## Perspective Transform using opencv and vertices calculated above
#     pts1 = np.float32([top_left, top_right, bottom_left, bottom_right])
#     pts2 = np.float32([[0,0],[width,0],[0,height],[width,height]])
#     matrix = cv2.getPerspectiveTransform(pts1,pts2)
#     image = cv2.warpPerspective(image,matrix,(width,height))
#     # show_image(image)

#     ######################################################################

#     return image

##################################################  Solution begins

def dist(p1, p2):
    return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def solution(image_path):
    image = cv2.imread(image_path)
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

    if dist(top_left, top_right)<36 or dist(bottom_right, bottom_left)<36: #if both are within 6 pixel of each other (in a respectable size image), then change slope as image is horizontal
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
    # score = ssim('Q1/ground_truth/1.png',image,channel_axis=-1)
    # print(score)

    ######################################################################

    return image


# def show_image(image):
#     cv2.imshow('image', image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#     # print("Done")

# ## Run function
# show_image(solution("Q1/test/4.png"))