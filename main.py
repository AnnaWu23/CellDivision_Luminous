# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
import random


def get_img_lis():
    # Use a breakpoint in the code line below to debug your script.
    for root, dirs, files in os.walk('./Sequences'):
        print('Root: ', root)
        print('Dirs : ', dirs)
        print('Files : ', files)


def plt_show_close():
    # cv2.waitKey(0)
    plt.show()
    plt.close()


def show_one_img(img_p):
    # img_path="./Sequences/02/t001.tif"
    img = cv2.imread(img_p)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb)
    # plt_show_close()

    f = plt.subplot(4, 3, 2)
    f.set_title(f'gray image', fontdict=plt_dic, loc='center', y=ypos)
    plt.xticks([])
    plt.yticks([])
    img_gray = cv2.imread(img_p, 0)
    # cv2.waitKey(0)

    plt.imshow(img_gray, 'gray', vmin=0, vmax=255)
    # cv2.waitKey()
    # plt_show_close()
    print(f'Contrast min = {np.min(img_gray)}\n'
          f'Contrast min = {np.max(img_gray)}\n'
          f'shape = {img_gray.shape}')
    return img_rgb, img_gray


# Press the green button in the gutter to run the script.
def Contrast_stretch(img):
    a, b = 0, 255
    c, d = np.min(img), np.max(img)
    img_o = (img - c) * ((b - a) / (d - c)) + a
    img_o = img_o.astype('uint8')
    plt.imshow(img_o, 'gray')
    # plt_show_close()
    return img_o


def binary(img):
    _, mask_otsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    plt.imshow(mask_otsu, 'gray')
    # plt_show_close()
    return mask_otsu


def erosion(img):
    # 腐蚀

    kernel = np.ones((2, 2), np.uint8)
    erosion = cv2.erode(img, kernel, iterations=1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(erosion, 'gray')
    # plt_show_close()


def dilation(img):
    # 膨胀
    # fig.set_title(f'dilation image',fontsize=11,fontweight='bold')

    kernel = np.ones((3, 3), np.uint8)
    dilation = cv2.dilate(img, kernel, iterations=1)
    # plt.figure((10,8),dpi=750)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(dilation, 'gray')
    return dilation
    # plt_show_close()


def Opening(img):
    # Opening is just another name of erosion followed by dilation.
    # It is useful in removing noise, as we explained above.
    # Here we use the function,
    # fig.set_title(f'Opening (erosion followed by dilation) image',fontsize=11,fontweight='bold')

    kernel = np.ones((2, 2), np.uint8)
    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)  # plt.figure((10,8),dpi=750)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(opening, 'gray')
    return opening
    # plt_show_close()


def find_countours(img):
    # #cell num count >30 pixels
    img_label, contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    area = []
    contours1 = []
    for i in contours:
        if cv2.contourArea(i) > 30:
            contours1.append(i)
    print(len(contours1) - 1)
    return contours1


def randomcolor():
    colorArr = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F']
    color = ""
    for i in range(6):
        color += colorArr[random.randint(0, 14)]
    return "#" + color


def draw_contours(img, tours):
    img_rgb_open = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    for e in tours:
        draw = cv2.drawContours(img_rgb_open, e, -1,
                                (
                                    random.randint(0, 255)
                                    , random.randint(0, 255)
                                    , random.randint(0, 255)
                                ), 3)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(draw)
    return draw


def mark_id(draw, tours):
    plt.xticks([])
    plt.yticks([])
    for i, j in zip(tours, range(len(tours))):
        M = cv2.moments(i)
        cX = int(M['m10'] / M["m00"])
        cY = int(M['m01'] / M["m00"])
        draw_id = cv2.putText(draw, str(j), (cX, cY), 1, 1, (255, 0, 255), 3)
    plt.imshow(draw_id)


if __name__ == '__main__':
    # get_img_lis()

    plt_dic = {'fontsize': 2,
               'fontweight': 'normal',
               'verticalalignment': 'top',
               'horizontalalignment': 'center',
               }
    plt.figure(figsize=(5, 1.8), dpi=750)
    img_path = "./Sequences/02/t001.tif"
    f = plt.subplot(4, 3, 1)
    plt.xticks([])
    plt.yticks([])
    ypos = 0.9
    f.set_title(f'cvtColor image', fontdict=plt_dic, loc='center',y=ypos)
    img_rgb, gray = show_one_img(img_path)

    f = plt.subplot(4, 3, 3)
    plt.xticks([])
    plt.yticks([])
    f.set_title(f'Contrast_stretch image', fontdict=plt_dic, loc='center', y=ypos)
    img_stretch = Contrast_stretch(gray)

    f = plt.subplot(4, 3, 4)
    plt.xticks([])
    plt.yticks([])
    f.set_title(f'binary image', fontdict=plt_dic, loc='center', y=ypos)
    img_binary = binary(img_stretch)
    # plt_show_close()

    f = plt.subplot(4, 3, 5)
    f.set_title(f'erosion image', fontdict=plt_dic, loc='center', y=ypos)
    erosion(img_binary)

    f = plt.subplot(4, 3, 6)
    f.set_title(f'dilation image', fontdict=plt_dic, loc='center', y=ypos)
    img_d = dilation(img_binary)

    f = plt.subplot(4, 3, 7)
    f.set_title(f'Opening image', fontdict=plt_dic, loc='center', y=ypos)
    img_open = Opening(img_binary)

    con_tours_e = find_countours(img_d)
    # draw connected area
    f = plt.subplot(4, 3, 8)
    f.set_title(f'drawContours on After erosion-dilation image', fontdict=plt_dic, loc='center', y=ypos)
    img_draw_e = draw_contours(img_d, con_tours_e)

    f = plt.subplot(4, 3, 10)
    f.set_title(f'Marking ID on ED img', fontdict=plt_dic, loc='center', y=ypos)
    mark_id(img_draw_e, con_tours_e)

    con_tours = find_countours(img_open)
    # draw connected area
    f = plt.subplot(4, 3, 9)
    f.set_title(f'drawContours on Opening', fontdict=plt_dic, loc='center', y=ypos)
    img_draw = draw_contours(img_open, con_tours)

    f = plt.subplot(4, 3, 11)
    f.set_title(f'Marking ID', fontdict=plt_dic, loc='center', y=ypos)
    mark_id(img_draw, con_tours)
    plt.savefig('./1-1result.png', dpi=750)
    plt_show_close()
#     细胞分裂 Detection
#     shape Google
#     求轮廓，内切圆，内切椭圆，
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
