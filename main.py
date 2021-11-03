import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

SEQUENCE = '01'
# If the cells' movement between two frame is less than DIST, it's the same cell.
DIST = 21
# The kernel size in opening.
OPEN_KERNEL_SIZE = 3
# The time for each image displaying when it is automatically playing.The unit is milliseconds.
SPEED = 500


images = []
# Data structure
# List images [
#   {
#       Image image_thre: the image after thresholding
#       Image image_open: the image after erosion and dilation
#       List<int> contours: A list storing the contours of cells in a frame
#       List<int> contours_center: A list saving the centroid of the cells in a frame
#       Image cell_track_draw: Images showing cells' contour, label, center and trajectories
#       Image cell_count_draw: Images showing the cells' contour, label, center, trajectories and count
#   },{...},{...}
# ]

cells_matching = {}
# Data structure
# Dict cells_matching {
#   int id: A unique number marking the cell {
#       Tuple center: (x, y) is the center of the cell
#       Tuple color: (x, y, z) is the code for RGB color
#       int index: the index of cell in list 'images'
#       List trajectories: the list of segments that cells move
#   },
#   int id:{...},
#   int id:{...}
# }

################## Task 1 ##################
# Collect the images from the whole directory
def image_read():
    image_path_list = []
    image_list = []
    # collect the images
    root = "Dataset/" + SEQUENCE
    files = os.listdir(root)
    files.sort()
    image_path_list = [str(root + "/" + img) for img in files]

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


# Stretching the image. Convert the 16 bit image to 8 bit image and augment the contrast
def image_stretch(image_list):
    output = []
    a = 0
    b = 255
    for img in image_list:
        c = np.min(img)
        d = np.max(img)
        image = ((img - c) * ((b - a) / (d - c)) + a).astype(np.uint8)
        output.append(image)

    # Check the image range
    check_image_range(output[0])

    # DEBUG: Uncomment to plot the image
    # plt.imshow(image_list[0], 'gray')
    # plt.show()
    return output


# Apply the threshold to the images - OTSU thresholding
def threshold(image_list):
    for img in image_list:
        _, image = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        dict = {'image_thre': image}
        images.append(dict)
        # DEBUG: Uncomment to see the images
        # cv2.imshow("threshold", image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

# Opening is just another name of erosion followed by dilation.
# It is useful in removing noise, as we explained above.
# Here we use the function,
# fig.set_title(f'Opening (erosion followed by dilation) image',fontsize=11,fontweight='bold')
def opening():
    for img in images:
        kernel = np.ones((OPEN_KERNEL_SIZE, OPEN_KERNEL_SIZE), np.uint8)
        opening = cv2.morphologyEx(img['image_thre'], cv2.MORPH_OPEN, kernel)  # plt.figure((10,8),dpi=750)
        img['image_open'] = opening
        # DEBUG: Uncomment to see the images
        # cv2.imshow("opening", opening)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # DEBUG: check the shape of the output
        # print("Type: " + str(opening.dtype) + " Shape: " + str(opening.shape))
        # gray = cv2.cvtColor(opening, cv2.COLOR_RGB2GRAY)?



# Segment cells in images, find the contours of the them, record the cells' contours label in the list 'images'
def contours():
    for img in images:
        contour, hierarchy = cv2.findContours(img['image_open'], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours_new = []
        for i in contour:
            if cv2.contourArea(i) > 30:
                contours_new.append(i)
        img['contours'] = contours_new


# Find the center of the cell, save it in the list 'images' and initializing the state of the cells by reading the
# cells' positions in the first image. Apply each cell a unique number and color.
def find_centroid():
    # Find the centroid for all images
    for img in images:
        centroid = []
        for i in img['contours']:
            M = cv2.moments(i)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            centroid.append((cX, cY))
        img['contours_center'] = centroid

    # Initialized the dictionary for the first image
    cells = images[0]['contours_center']
    for cell, i in zip(cells, range(0, len(cells))):
        dict = {'center': cell, 'color': color_generator(), 'index': i, 'trajectories': []}
        cells_matching[i] = dict

    # Draw the contours, center and label for the image
    images[0]['cell_track_draw'] = images[0]['image_thre']
    draw_contours_center_label(0)

    # DEBUG
    # plt.imshow(images[0]['contours_center_draw'])
    # plt.show()


# Label the cells for each image
def label_cells():
    global cells_matching
    # Loop through from the second image
    for index in range(1, len(images)):
        new_cells_matching = {}
        # Scan through the cells in the unlabeled image 'images[index]', label it as we need
        centers_new = images[index]['contours_center']
        for cell_center, i in zip(centers_new, range(len(centers_new))):
            inherent_id = cell_in_last_image(cell_center)
            if inherent_id is not -1:
                # The cell still exists, update the position of the cell in dictionary cells_matching
                cell_old = cells_matching.pop(inherent_id)
                trajectories = cell_old['trajectories'] + [(cell_center, cell_old['center'])]
                new_cells_matching[inherent_id] = {'center': cell_center, 'color': cell_old['color'],
                                                   'index': i, 'trajectories': trajectories}
            else:
                # conditions: cells jumping...
                # Give the cell a new label, delete the nearest label from the last frame
                max_id_new = 0 if len(new_cells_matching) is 0 else max(new_cells_matching)
                max_id_old = 0 if len(cells_matching) is 0 else max(cells_matching)
                max_id = max(max_id_old, max_id_new)
                new_cells_matching[max_id + 1] = {'center': cell_center, 'color': color_generator(),
                                                  'index': i, 'trajectories': []}

        # DEBUG: Check if the number of cells detected is same to what we saved in dictionary
        # if len(new_cells_matching) == len(centers_new):
        #     print("yes")
        # else:
        #     print('no')

        cells_matching = new_cells_matching
        # Draw the contours, center and label for the image
        draw_contours_center_label(index)


def cell_count():
    global cells_matching, images
    for frame in images:
        count = cv2.putText(frame['cell_track_draw'], "COUNT: " + str(len(frame['contours'])),
                            (50, 50), 1, 2, (255, 0, 255), 2)
        frame['cell_count_draw'] = count


###################### HELPER FUNCTION ######################
# Print out the image's max and min value of the pixels
def check_image_range(img):
    minPixel = np.min(img)
    maxPixel = np.max(img)
    print("The minimum value of the image: " + str(minPixel))
    print("The maximum value of the image: " + str(maxPixel))


# Generating an RGB color without overlapping
def color_generator():
    global cells_matching
    # generate a number randomly
    while True:
        color = (random.randint(0, 256), random.randint(0, 256), random.randint(0, 256))
        if not any(cell['color'] == color for cell in cells_matching.values()):
            return color


# Print out the histogram of the particular image
def show_histogram(img):
    plt.hist(img.ravel(), 256, [0, 256])
    plt.show()


# It takes the cell's center axis and the cells' position from the last frame, check if the cell is still
# exist in this frame.
# return: Corresponding id if the cell still exists, otherwise -1
def cell_in_last_image(cell):
    for id, old_cell in cells_matching.items():
        if old_cell['center'][0] - DIST < cell[0] <= old_cell['center'][0] + DIST \
                and old_cell['center'][1] - DIST < cell[1] <= old_cell['center'][1] + DIST:
            return id
    return -1


# Automatically playing the image
def display_all_images(image_list):
    for img in image_list:
        cv2.imshow("image", img)
        cv2.waitKey(SPEED)
        cv2.destroyAllWindows()


# Take the image list and save it
def save_images(image_list, dir):
    for img, i in zip(image_list, range(len(image_list))):
        cv2.imwrite(dir + "/" + SEQUENCE + "/" + str(i) + ".png", img)


# Draw the contours for cells in terms of the current cells_matching
def draw_contours_center_label(images_index):
    global cells_matching, images
    img_thre = images[images_index]['image_thre']
    img_thre = cv2.cvtColor(img_thre, cv2.COLOR_GRAY2RGB)
    draw_trajectories = img_thre
    for id, cell in cells_matching.items():
        draw_contours = cv2.drawContours(img_thre, images[images_index]['contours'],
                                         cell['index'], cells_matching[id]['color'], 2)
        draw_center = cv2.circle(draw_contours, cell['center'], 0, cells_matching[id]['color'], 2)
        draw_label = cv2.putText(draw_center, str(id), cell['center'], 1, 1, (255, 0, 255), 2)
        for segment in cell['trajectories']:
            draw_trajectories = cv2.line(draw_label, segment[0], segment[1],
                                         cells_matching[id]['color'], thickness=2)
    images[images_index]['cell_track_draw'] = draw_trajectories



# NOTICE: WE USE OPENING INSTEAD OF EROSION THEN DILATION
# # Dealing with image erosion, target to reducing the noise in the threshold image
# def erosion(img):
#     kernel = np.ones((3, 3), np.uint8)
#     erosion = cv2.erode(img, kernel, iterations=1)
#     cv2.imshow("erosion", erosion)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#     return erosion
#
#
# # Dealing with the image dilation, roll back the cell to the original size
# def dilation(img):
#     # fig.set_title(f'dilation image',fontsize=11,fontweight='bold')
#     kernel = np.ones((3, 3), np.uint8)
#     dilation = cv2.dilate(img, kernel, iterations=1)
#     cv2.imshow("dilation", dilation)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#     return dilation


if __name__ == '__main__':
    # Preprocess
    # a. Read the images
    image_list = image_read()
    # b. Check the image range
    check_image_range(image_list[0])
    # c. Since the image contrast is too low, let's do a contrast stretch first
    image_list = image_stretch(image_list)
    # save_images(image_list, "Dataset/AllImagesAfterStretching")
    # d. No equalization needed, UNCOMMENT THIS TO CHECK THE HISTOGRAM
    # show_histogram(image_list[0])
    # e. Thresholding
    threshold(image_list)
    # save_images([img['image_thre'] for img in images], "Dataset/AllImagesAfterThreshold")
    # f. remove the noise of the images by erosion and dilation
    opening()
    # save_images([img['image_open'] for img in images], "Dataset/AllImagesAfterOpening")
    # Task 1.1: Segment all the cells and show their contours in the images as overlays.
    contours()
    # display_all_images([img['cell_track_draw'] for img in images])
    # Task 1.2: Track all the cells over time and show their trajectories as overlays.
    # a. Find the center of the cells, label it and save the diagram
    find_centroid()
    # b. Loop through all the frame, recognise the same cell and label it, label the trajectories at the same time
    label_cells()
    # display_all_images([img['cell_track_draw'] for img in images])
    # save_images([img['cell_track_draw'] for img in images], "Dataset/AllImagesWithTrajectories")
    # Task 2.1: The cell count (the number of cells) in the image.
    cell_count()
    display_all_images([img['cell_count_draw'] for img in images])

    # save_images([img['cell_track_draw'] for img in images], "Dataset/AllImagesWithTrajectories")

    # Task 2.2: The average size (in pixels) of all the cells in the image.
    # Task 2.3: The average displacement (in pixels) of all the cells, from the previous image to the
    #           current image in the sequence.
    # Task 2.4: The number of cells that are in the process of dividing. visually alert the viewer
    #           where in the image these divisions are happening
