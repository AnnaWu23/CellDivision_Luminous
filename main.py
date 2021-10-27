import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

################## Task 1 ##################
# Collect the images from the whole directory
def image_read(path):
    image_path_list = []
    image_list = []

    # collect the non-GT images
    for root, dirs, files in os.walk(path):
        if "GT" not in root and len(files) > 3:
            image_path_list += [str(root + "\\" + img) for img in files]

    # DEBUG: Uncomment this to check the path name
    # for img in image_path_list:
    #     print(img)

    # Check the image shape
    img_BGR = cv2.imread(image_path_list[0])
    img_RGB = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)
    print("The shape of the RGB image: " + str(img_RGB.shape))
    print("Check three channel: " + str(img_RGB[0][0]))

    # As the value in three channel is same, we read it as a 16-bit gray image
    for image_path in image_path_list:
        img_16bit_gray = cv2.imread(image_path, cv2.IMREAD_ANYDEPTH)
        image_list.append(img_16bit_gray)

    # DEBUG: Uncomment to plot the image
    # plt.imshow(image_list[0], 'gray')
    # plt.show()

    return image_list

# Print out the image's max and min value of the pixels
def check_image_range(img):
    minPixel = np.min(img)
    maxPixel = np.max(img)
    print("The minimum value of the image: " + str(minPixel))
    print("The maximum value of the image: " + str(maxPixel))

def image_stretch(image_list):
    a = 0
    b = 255
    for i in range(len(image_list)):
        c = np.min(image_list[i])
        d = np.max(image_list[i])
        image_list[i] = ((image_list[i] - c) * ((b - a) / (d - c)) + a).astype(np.uint8)

    # Check the image range
    check_image_range(image_list[0])

    # DEBUG: Uncomment to plot the image
    # plt.imshow(image_list[0], 'gray')
    # plt.show()

# Print out the histogram of the particular image
def show_histogram(img):
    plt.subplot(2, 2, 2), plt.hist(img.ravel(), 256, [0, 256])
    plt.show()

def threshold(image_list):
    for i in range(len(image_list)):
        _, image_list[i] = cv2.threshold(image_list[i], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

def display_all_images(image_list):
    for img in image_list:
        cv2.imshow("image", img)
        cv2.waitKey(500)
        cv2.destroyAllWindows()

if __name__ == '__main__':
    # Read the images
    image_list = image_read(".\Dataset")
    # Check the image range
    check_image_range(image_list[0])
    # Since the image contrast is too low, let's do a contrast stretch first
    image_stretch(image_list)
    # No equalization needed
    show_histogram(image_list[0])
    # Thresholding
    threshold(image_list)
    # display_all_images(image_list)