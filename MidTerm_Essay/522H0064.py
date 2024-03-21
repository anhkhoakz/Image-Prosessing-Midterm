import cv2 as cv
import numpy as np


def task1():
    frame = cv.imread("input1.jpg")
    frame_gray = cv.imread("input1.jpg", cv.IMREAD_GRAYSCALE)

    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    color_dict_HSV = {
        "yellow": [[35, 255, 255], [27, 50, 70]],
        "orange": [[24, 255, 255], [8, 140, 200]],
        "red": [[180, 255, 255], [159, 50, 70]],
        "blue": [[128, 255, 255], [90, 50, 70]],
        "green": [[89, 255, 255], [36, 50, 70]],
        "purple": [[158, 255, 255], [129, 50, 70]],
    }

    for key in color_dict_HSV:
        lower = np.array(color_dict_HSV[key][1])
        upper = np.array(color_dict_HSV[key][0])
        mask = cv.inRange(hsv, lower, upper)
        balloon = cv.bitwise_and(frame, frame, mask=mask)
        cv.imwrite("./Task1_Result/" + str(key) + ".png", balloon)

    _, thresh = cv.threshold(frame_gray, 225, 255, cv.THRESH_TOZERO)
    kernel = np.ones((27, 27), np.uint8)
    dillation = cv.dilate(thresh, kernel, iterations=0)

    contours, _ = cv.findContours(
        image=dillation, mode=cv.RETR_TREE, method=cv.CHAIN_APPROX_NONE
    )

    for cnt in contours:
        cv.drawContours(
            image=frame_gray,
            contours=[cnt],
            contourIdx=0,
            color=255,
            thickness=cv.FILLED,
        )

    _, dst = cv.threshold(frame_gray, 254, 255, cv.THRESH_BINARY_INV)

    cv.imwrite("./Task1_Result/Task1b.png", dst)


def task2():
    img = cv.imread("input2.png")
    img_gray = cv.imread("input2.png", 0)

    img_edit_1 = img[494:558, 263:300]
    img_edit_2 = img[494:558, 300:369]
    img_edit_3 = img[395:474, 288:437]
    img_edit_5 = img[207:263, 367:395]
    img_edit_6 = img[299:365, 334:373]
    img_edit_7 = img[302:368, 430:472]
    img_edit_8 = img[493:560, 365:408]
    img_edit_9 = img[500:567, 419:469]
    img_edit_10 = img[300:378, 468:500]

    th1 = cv.adaptiveThreshold(
        img_gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 11, 2
    )
    contours, _ = cv.findContours(th1, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    min_contour_area = 150
    max_countour_area = 1450

    for cnt in contours:
        area = cv.contourArea(cnt)
        if area > min_contour_area and area < max_countour_area:
            x, y, w, h = cv.boundingRect(cnt)
            cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    def edit(img, iteration, kernels, lower):
        kernel = np.ones((kernels, kernels), np.uint8)
        closing = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel)

        closing_gray = cv.cvtColor(closing, cv.COLOR_BGR2GRAY)

        _, thresh = cv.threshold(closing_gray, lower, 255, cv.THRESH_BINARY)
        dillation = cv.erode(thresh, kernel, iterations=iteration)

        contours1, _ = cv.findContours(dillation, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
        min_contour_area = 350
        max_countour_area = 1600

        for cnt in contours1:
            area = cv.contourArea(cnt)
            if area > min_contour_area and area < max_countour_area:
                x, y, w, h = cv.boundingRect(cnt)
                cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        return img

    img_edit_1 = edit(img_edit_1, 1, 2, 1)
    img_edit_2 = edit(img_edit_2, 1, 3, 1)
    img_edit_3 = edit(img_edit_3, 1, 3, 30)
    img_edit_5 = edit(img_edit_5, 2, 1, 1)
    img_edit_6 = edit(img_edit_6, 1, 2, 10)
    img_edit_7 = edit(img_edit_7, 3, 2, 10)
    img_edit_8 = edit(img_edit_8, 2, 3, 1)
    img_edit_9 = edit(img_edit_9, 2, 2, 1)
    img_edit_10 = edit(img_edit_10, 2, 3, 10)

    cv.imwrite("./Task2_Result/task2.png", img)


task1()
task2()
