import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

images = []
# Data structure
# List images [
#   {
#       Image image_thre: the image after thresholding
#       List<int> contours: A list storing the contours of cells in a frame
#       List<int> contoursCenter: A list saving the centroid of the contours
#       Image contoursDraw: the images showing the contours and the label
#       Image contoursCenterDraw: the images showing the contour and it's center
#       Image cellTrackDraw: Images showing the trajectories as well as the contours
#   },{...},{...}
# ]

cells_matching = {}
# Data structure
# List cells_matching {
#   int id: A unique number marking the cell {
#       Tuple center: (x, y) is the center of the cell
#       Tuple color: (x, y, z) is the code for RGB color
#       int index: the index of cell in list images
#       List trajectories: the list of segments that cells move
#   },
#   int id:{...},
#   int id:{...}
# }

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
    check_image_range(image_list[0])

    # DEBUG: Uncomment to plot the image
    # plt.imshow(image_list[0], 'gray')
    # plt.show()
    return output


# Print out the histogram of the particular image
def show_histogram(img):
    plt.subplot(2, 2, 2), plt.hist(img.ravel(), 256, [0, 256])
    plt.show()


# Apply the threshold to the images - OTSU thresholding
def threshold(image_list):
    output = []
    for img in image_list:
        _, image = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        output.append(image)
    return output


# Automatically playing the image
def display_all_images(image_list):
    for img in image_list:
        cv2.imshow("image", img)
        cv2.waitKey(500)
        cv2.destroyAllWindows()


# Find the contours of the images, record the cells' contours label in the list 'images', record the thresholding images
# in the list 'images'
def contours(image_list):
    for img in image_list:
        img_label, contour, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours_new = []
        for i in contour:
            if cv2.contourArea(i) > 30:
                contours_new.append(i)
        imgO = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        dict = {'contours': contours_new, 'image_thre': imgO}
        images.append(dict)


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
        img['contoursCenter'] = centroid

    # Initialized the dictionary for the first image
    cells = images[0]['contoursCenter']
    for cell, i in zip(cells, range(0, len(cells))):
        dict = {'center': cell, 'color': color_generator(), 'index': i, 'trajectories': []}
        cells_matching[i] = dict

    # Draw the contours, center and label for the image
    images[0]['cellTrackDraw'] = images[0]['image_thre']
    draw_contours_center_label(0)

    # DEBUG
    # plt.imshow(images[0]['contoursCenterDraw'])
    # plt.show()


# Generating an RGB color without overlapping
def color_generator():
    global cells_matching
    # generate a number randomly
    while True:
        color = (random.randint(0, 256), random.randint(0, 256), random.randint(0, 256))
        if not any(cell['color'] == color for cell in cells_matching.values()):
            return color


# Draw the contours for cells in terms of the current cells_matching
def draw_contours_center_label(images_index):
    global cells_matching, images
    draw_contours = images[images_index]['image_thre']
    draw_center = draw_contours
    draw_label = draw_contours
    draw_trajectories = draw_contours
    for id, cell in cells_matching.items():
        draw_contours = cv2.drawContours(draw_contours, images[images_index]['contours'],
                                         cell['index'], cells_matching[id]['color'], 2)
        draw_center = cv2.circle(draw_contours, cell['center'], 0, cells_matching[id]['color'], 2)
        draw_label = cv2.putText(draw_center, str(id), cell['center'], 1, 1, (255, 0, 255), 2)
        for segment in cell['trajectories']:
            draw_trajectories = cv2.line(draw_label, segment[0], segment[1],
                         cells_matching[id]['color'], thickness=2)
    images[images_index]['contoursDraw'] = draw_contours
    images[images_index]['coantoursCenterDraw'] = draw_label
    images[images_index]['cellTrackDraw'] = draw_trajectories

# Label the cells for each image
def label_cells():
    global cells_matching
    # Loop through from the second image
    for index in range(1, len(images)):
        new_cells_matching = {}
        # Scan through the cells in the unlabeled image 'images[index]', label it as we need
        centers_new = images[index]['contoursCenter']
        for cell_center, i in zip(centers_new, range(len(centers_new))):
            inherent_id = cell_in_last_image(cell_center)
            if inherent_id is not -1:
                # The cell still exists, update the position of the cell in dictionary cells_matching
                cell_old = cells_matching.pop(inherent_id)
                trajectories = cell_old['trajectories'] + [(cell_center, cell_old['center'])]
                new_cells_matching[inherent_id] = {'center': cell_center, 'color': cell_old['color'],
                                               'index': i, 'trajectories': trajectories}
            else:
                # TODO: check if the cell is dividing or not
                # conditions: cells jumping...
                # Give the cell a new label, delete the nearest label from the last frame
                # delete_possible_cell(cell_center)
                max_id_new = 0 if len(new_cells_matching) is 0 else max(new_cells_matching)
                max_id_old = 0 if len(cells_matching) is 0 else max(cells_matching)
                max_id = max(max_id_old, max_id_new)
                new_cells_matching[max_id + 1] = {'center': cell_center, 'color': color_generator(),
                                                  'index': i, 'trajectories': []}

        # if len(new_cells_matching) == len(centers_new):
        #     print("yes")
        # else:
        #     print('no')
        cells_matching = new_cells_matching
        # Draw the contours, center and label for the image
        draw_contours_center_label(index)


# It takes the cell's center axis and the cells' position from the last frame, check if the cell is still
# exist in this frame.
# return: Corresponding id if the cell still exists, otherwise -1
def cell_in_last_image(cell):
    dist = 21
    for id, old_cell in cells_matching.items():
        if old_cell['center'][0] - dist < cell[0] <= old_cell['center'][0] + dist \
                and old_cell['center'][1] - dist < cell[1] <= old_cell['center'][1] + dist:
            return id
    return -1


# In terms of the cell's center given, delete the cells, which is the most likely to disappear ,from the
# list 'cells_matching'
def delete_possible_cell(new_cell):
    nearest_dist = float("inf")
    nearest_cell_id = 0
    for id, old_cell in cells_matching.items():
        dist = ((old_cell['center'][0] - new_cell[0]) ** 2 + (old_cell['center'][1] - new_cell[1]) ** 2) ** 0.5
        if dist < nearest_dist:
            nearest_dist = dist
            nearest_cell_id = id
    cells_matching.pop(nearest_cell_id)


if __name__ == '__main__':
    # Prepare
    # Read the images
    image_list = image_read(".\Dataset")
    # Check the image range
    check_image_range(image_list[0])
    # Since the image contrast is too low, let's do a contrast stretch first
    image_list = image_stretch(image_list)
    # No equalization needed
    # show_histogram(image_list[0])
    # Thresholding
    image_list = threshold(image_list)
    # Task 1.1: Segment all the cells and show their contours in the images as overlays.
    # TODO: ADD YOUR CODE HERE
    # Task 1.2: Track all the cells over time and show their trajectories as overlays.
    # There are two aspect of this task: 1. cell tracking, 2. cell dividing detecting
    # Cell tracking
    # a. Label the centroid position
    # Find the contours in the image
    contours(image_list)
    # Find the center of the contours
    find_centroid()
    # Loop through all the frame, recognise the same cell and label it
    label_cells()
    display_all_images([img['cellTrackDraw'] for img in images])
    # b. Associate each cell in any frame to the spatially nearest cell in the next frame within a predefined range

