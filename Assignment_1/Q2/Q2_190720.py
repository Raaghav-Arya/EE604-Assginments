import cv2
import numpy as np

# np.set_printoptions(threshold=np.inf)
import librosa as lr
# import matplotlib.pyplot as plt



# def show_plot(matrix):
#     plt.imshow(matrix, aspect="auto")
#     plt.colorbar()
#     plt.show()


# def show_image(matrix):
#     plt.imshow(matrix, cmap="gray")
#     plt.show()
#     # print("Done")



def solution(audio_path):
    ############################
    ############################

    y, sr = lr.load(audio_path, sr=None)

    n_fft = 2048
    hop_length = 512

    mat = lr.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, fmax=8000
    )
    # mat = lr.power_to_db(mat)
    # show_plot(mat)

    thresh_val, thresh = cv2.threshold(mat, 10, 255, cv2.THRESH_BINARY)
    # show_image(thresh)

    # print(thresh)

    weighted_sum = 0
    pts = 0

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if thresh[i][j] == 255:
                weighted_sum += i**2
                pts += 1

    weighted_mean = weighted_sum / pts
    # print(weighted_mean)

    if weighted_mean > 1200:
        return "metal"
    else:
        return "cardboard"

    ## AMPLITUDE PLOT
    # plt.plot(y)
    # plt.show()

    ############################
    ############################
    ## comment the line below before submitting else your code wont be executed##
    # pass
    # class_name = 'cardboard'
    # return class_name



# print(solution("Q2/test/cardboard1.mp3"))


# print(solution("Q2/test/cardboard2.mp3"))


# print(solution("Q2/test/metal_banging1.mp3"))


# print(solution("Q2/test/metal_banging2.mp3"))
