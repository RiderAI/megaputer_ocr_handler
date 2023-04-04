import cv2
import numpy


def perspective_correction(img: numpy.ndarray = None, dots: list = [[100, 260], [700, 260], [0, 800], [1100, 800]]) -> numpy.ndarray:
    # Locate points of the documents or object which you want to transform
    pts1 = numpy.float32(dots)
    pts2 = numpy.float32([[0, 0], [1000, 0],
                          [0, 1400], [1000, 1400]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    result = cv2.warpPerspective(img, matrix, (1000, 1400))
    return result


def draw_polyline(img: numpy.ndarray = None, dots: list = [[100, 260], [700, 260], [1100, 800], [0, 800]]) -> numpy.ndarray:
    pts = numpy.int32(dots)
    is_closed = True
    color = (255, 0, 0)  # Blue color in BGR
    thickness = 2  # Line thickness of 2 px
    # Using cv2.polylines() method draw a Blue polygon with thickness of 2 px
    result = cv2.polylines(img, [pts], is_closed, color, thickness)
    return result


def get_edges(img: numpy.ndarray = None) -> numpy.ndarray:
    # Перевести изображение в оттенки серого
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Найти границы изображения с помощью Canny edge detector
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    result = edges
    return result
