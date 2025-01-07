# utlis.py

import cv2
import numpy as np

def biggestContour(contours, min_area=5000):
    """
    Tìm contour lớn nhất có diện tích lớn hơn min_area và có 4 điểm.

    Args:
        contours (list): Danh sách các contour.
        min_area (int): Diện tích tối thiểu để xem xét.

    Returns:
        tuple: Contour lớn nhất và diện tích của nó.
    """
    biggest = np.array([])
    max_area = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > min_area:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area
    return biggest, max_area

def reorder(points):
    """
    Sắp xếp lại các điểm theo thứ tự: trên-trái, trên-phải, dưới-phải, dưới-trái.

    Args:
        points (np.ndarray): Mảng các điểm.

    Returns:
        np.ndarray: Mảng các điểm đã sắp xếp.
    """
    points = points.reshape((4, 2))
    new_points = np.zeros((4, 1, 2), dtype=np.int32)
    add = points.sum(1)
    new_points[0] = points[np.argmin(add)]
    new_points[3] = points[np.argmax(add)]
    diff = np.diff(points, axis=1)
    new_points[1] = points[np.argmin(diff)]
    new_points[2] = points[np.argmax(diff)]
    return new_points

def drawRectangle(img, biggest, thickness=2):
    """
    Vẽ hình chữ nhật quanh contour lớn nhất.

    Args:
        img (np.ndarray): Ảnh cần vẽ.
        biggest (np.ndarray): Contour lớn nhất.
        thickness (int): Độ dày của đường viền.

    Returns:
        np.ndarray: Ảnh đã được vẽ hình chữ nhật.
    """
    cv2.line(img, tuple(biggest[0][0]), tuple(biggest[1][0]), (0, 255, 0), thickness)
    cv2.line(img, tuple(biggest[1][0]), tuple(biggest[2][0]), (0, 255, 0), thickness)
    cv2.line(img, tuple(biggest[2][0]), tuple(biggest[3][0]), (0, 255, 0), thickness)
    cv2.line(img, tuple(biggest[3][0]), tuple(biggest[0][0]), (0, 255, 0), thickness)
    return img

def stackImages(imgArray, scale, labels=None):
    """
    Xếp chồng các ảnh để hiển thị.

    Args:
        imgArray (list): Danh sách các ảnh cần xếp chồng.
        scale (float): Tỷ lệ kích thước.
        labels (list, optional): Danh sách các nhãn cho từng ảnh.

    Returns:
        np.ndarray: Ảnh đã được xếp chồng.
    """
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range(len(imgArray)):
            for y in range(len(imgArray[x])):
                imgArray[x][y] = cv2.resize(imgArray[x][y], (0,0), fx=scale, fy=scale)
                if len(imgArray[x][y].shape) == 2:
                    imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
                if labels and len(labels[x]) > y:
                    cv2.putText(imgArray[x][y], labels[x][y], (10, height - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        for x in range(rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            imgArray[x] = cv2.resize(imgArray[x], (0,0), fx=scale, fy=scale)
            if len(imgArray[x].shape) == 2:
                imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
            if labels and len(labels) > x:
                cv2.putText(imgArray[x], labels[x], (10, height - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        hor = np.hstack(imgArray)
        ver = hor
    return ver
