import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import filters
import cv2


def disparitySSD(img_l: np.ndarray, img_r: np.ndarray, disp_range: (int, int), k_size: int) -> np.ndarray:
    """
    img_l: Left image
    img_r: Right image
    range: Minimum and Maximum disparity range. Ex. (10,80)
    k_size: Kernel size for computing the SSD, kernel.shape = (k_size*2+1,k_size*2+1)

    return: Disparity map, disp_map.shape = Left.shape
    """
    # Links I used:
    # https://pramod-atre.medium.com/disparity-map-computation-in-python-and-c-c8113c63d701#:~:text=Depth%20is%20inversely%20proportional%20to,we%20would%20have%20higher%20Z
    # Remember:
    # SSD- we are taking patches in the image,decrease, square it and sum all- the minimum value we get will be our chosen value.

    # We will take the size of the image
    height, width = img_r.shape
    # Define max disparity range
    max_offset = disp_range[1]
    # Create new zeros array when the dimensions represented by(height,width,maximun disparity range(i.e. depth of the maximum offset))
    disparity_map = np.zeros((height, width, max_offset))
    # Define new arrays of zeros in the same size as img_r
    average_left = np.zeros((height, width))
    average_right = np.zeros((height, width))

    # Compute average of our window (i.e. our image kernel) by using uniform_filter
    filters.uniform_filter(img_l, k_size, average_left)
    filters.uniform_filter(img_r, k_size, average_right)
    # Now we will normalize left and right images
    # Normalized left image
    normalized_left = img_l - average_left
    # Normalized right image
    normalized_right = img_r - average_right
    # For each offset in the given range
    for i in range(disp_range[1]):
        # moving i element(from the end) of the normalized right image to the front (i.e. to be the first element)
        right_img_shift = np.roll(normalized_right, i)
        # Normalization
        filters.uniform_filter(normalized_left * right_img_shift, k_size, disparity_map[:, :, i])
        # update disparity_map with SSD score
        disparity_map[:, :, i] = disparity_map[:, :, i] ** 2  # (Li-Ri)^2
    # for each pixel we will choose the best (i.e. maximum) depth value
    ans = np.argmax(disparity_map, axis=2)
    return ans


def disparityNC(img_l: np.ndarray, img_r: np.ndarray, disp_range: int, k_size: int) -> np.ndarray:
    """
    img_l: Left image
    img_r: Right image
    range: The Maximum disparity range. Ex. 80
    k_size: Kernel size for computing the NormCorolation, kernel.shape = (k_size*2+1,k_size*2+1)

    return: Disparity map, disp_map.shape = Left.shape
    """
    # First part is same as before:
    # We will take the size of the image
    height, width = img_r.shape
    # Define max disparity range
    max_offset = disp_range[1]
    # Create new zeros array when the dimensions represented by(height,width,maximum disparity range(i.e. depth of the maximum offset))
    disparity_map = np.zeros((height, width, max_offset))
    # Define new arrays of zeros in the same size as img_r
    average_left = np.zeros((height, width))
    average_right = np.zeros((height, width))

    # Compute average of our window (i.e. our image kernel) by using uniform_filter
    filters.uniform_filter(img_l, k_size, average_left)
    filters.uniform_filter(img_r, k_size, average_right)
    # Now we will normalize left and right images
    # Normalized the left image
    normalized_left = img_l - average_left
    # Normalized the right image
    normalized_right = img_r - average_right

    # Now we will Define the sigma's matrices in the same size of the right image
    sigma_left = np.zeros((height, width))
    sigma_right = np.zeros((height, width))
    sigma = np.zeros((height, width))
    # Compute the average of each pixel in window of size:  (k_size)^2
    filters.uniform_filter(normalized_left * normalized_left, k_size, sigma_left)

    # For each offset in the given range
    for i in range(disp_range[1]):
        # Moving i - disp_range[0] element(from the end) of the normalized right image to the front (i.e. to be the first element)
        right_img_shift = np.roll(normalized_right, i - disp_range[0])
        # Compute the sigma by using uniform_filter
        filters.uniform_filter(normalized_left * right_img_shift, k_size, sigma)
        # Compute the sigma_right by using uniform_filter
        filters.uniform_filter(right_img_shift * right_img_shift, k_size, sigma_right)

        sqr_sigma = np.sqrt(sigma_right * sigma_left)
        # Update the disparity_map with sigma
        disparity_map[:, :, i] = sigma / sqr_sigma
    # for each pixel we will choose the best (i.e. maximum) depth value
    ans = np.argmax(disparity_map, axis=2)
    return ans


def computeHomography(src_pnt: np.ndarray, dst_pnt: np.ndarray) -> (np.ndarray, float):
    """
    Finds the homography matrix, M, that transforms points from src_pnt to dst_pnt.
    returns the homography and the error between the transformed points to their
    destination (matched) points. Error = np.sqrt(sum((M.dot(src_pnt)-dst_pnt)**2))

    src_pnt: 4+ keypoints locations (x,y) on the original image. Shape:[4+,2]
    dst_pnt: 4+ keypoints locations (x,y) on the destination image. Shape:[4+,2]

    return: (Homography matrix shape:[3,3], Homography error)
    """
    # Links I used:
    # Presentation number 7, DLT algorithm, page number 30-34
    # https://www.youtube.com/watch?v=l_qjO4cM74o
    # Define A matrix
    A = []
    for i in range(len(src_pnt)):
        # Define the src vector
        x, y = src_pnt[i][0], src_pnt[i][1]
        # Define the dest vector
        u, v = dst_pnt[i][0], dst_pnt[i][1]
        # init A matrix (A is (2n x 9) mat)
        # like we learned in class, append for each point two rows below:
        A.append([x, y, 1, 0, 0, 0, -u * x, -u * y, -u])
        A.append([0, 0, 0, x, y, 1, -v * x, -v * y, -v])

    # Use SVD to find the values of the variables in the matrix
    U, S, Vh = np.linalg.svd(np.asarray(A))
    # Divided by the last row like we see in the exercise and reshaping to 3 by 3 (i.e. isolate H matrix and normalize)
    homography_mat = (Vh[-1, :] / Vh[-1, -1]).reshape(3, 3)

    # Compute the error
    homography_err = 0.
    for j in range(len(src_pnt)):
        src = np.append(src_pnt[j], 1)
        dst = np.append(dst_pnt[j], 1)
        # According to page number 36 in presentation number 7
        # And according for given formula in the function:  Error = np.sqrt(sum((M.dot(src_pnt)-dst_pnt)**2)
        homography_err = np.sqrt(sum(homography_mat.dot(src) / homography_mat.dot(src)[-1] - dst) ** 2)

    # Self checking with OpenCV function
    # homography_mat_cv, _ = cv2.findHomography(src_pnt, dst_pnt)
    # print('Homography Mat', homography_mat_cv)

    return homography_mat, homography_err


def warpImag(src_img: np.ndarray, dst_img: np.ndarray) -> None:
    """
    Displays both images, and lets the user mark 4 or more points on each image.
    Then calculates the homography and transforms the source image on to the destination image.
    Then transforms the source image onto the destination image and displays the result.

    src_img: The image that will be ’pasted’ onto the destination image.
    dst_img: The image that the source image will be ’pasted’ on.

    output: None.
    """
    # Sources I used:
    # Lecture number 7, page number 39
    # page number 21: http://www.cs.columbia.edu/~allen/F17/NOTES/homography_pka.pdf

    dst_p = []
    fig1 = plt.figure()

    # FIRST STEP:Displays both images, and lets the user mark 4 or more points on each image.
    # Mark 4 points on the dst image
    def onclick_1(event):
        x = event.xdata
        y = event.ydata
        print("Loc: {:.0f},{:.0f}".format(x, y))

        plt.plot(x, y, '*r')
        dst_p.append([x, y])

        if len(dst_p) == 4:
            plt.close()
        plt.show()

    # display image 1
    cid = fig1.canvas.mpl_connect('button_press_event', onclick_1)
    plt.imshow(dst_img)
    plt.show()
    dst_p = np.array(dst_p)

    ##### Your Code Here ######
    # We will do exact like written above,but for the second image

    src_p = []  # define array of the point we will mark
    fig2 = plt.figure()

    # Mark 4 points on the src image
    def onclick_2(event):
        x = event.xdata
        y = event.ydata
        print("Loc: {:.0f},{:.0f}".format(x, y))

        plt.plot(x, y, '*r')
        src_p.append([x, y])

        if len(src_p) == 4:
            plt.close()
        plt.show()

    # display image 2
    cid_dest = fig2.canvas.mpl_connect('button_press_event', onclick_2)
    plt.imshow(src_img)
    plt.show()
    src_p = np.array(src_p)

    # SECOND STEP: Calculates the homography
    homography, error = computeHomography(src_p, dst_p)

    # THIRD STEP: Transforms the source image on to the destination image. Then transforms the source image onto the destination image and displays the result.
    # Iterate over all pixels in src_img
    for Y_index in range(src_img.shape[0]):
        for X_index in range(src_img.shape[1]):
            # Define [x,y,1] as we learned in homographies (i.e., projective transformations)
            mat_homog = np.array([X_index, Y_index, 1])
            # Inner multiplication between homography matrix and [Xi, Yi, 1]
            mat_homog = homography.dot(mat_homog)
            # Divided by the second row
            mat_homog /= mat_homog[2]
            # Put the new pixels in the dst image
            dst_img[int(mat_homog[1]), int(mat_homog[0])] = src_img[Y_index, X_index]
    # Display the new stitches image
    plt.imshow(dst_img)
    plt.show()
