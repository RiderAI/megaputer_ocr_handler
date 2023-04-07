import math
import cv2
import numpy as np
import easyocr
from deprecated import deprecated


def perspective_correction(img: np.ndarray = None, dots: list = [[100, 260], [700, 260], [0, 800], [1100, 800]]) -> np.ndarray:
    # Locate points of the documents or object which you want to transform
    pts1 = np.float32(dots)
    pts2 = np.float32([[0, 0], [1000, 0],
                       [0, 1400], [1000, 1400]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    result = cv2.warpPerspective(img, matrix, (1000, 1400))
    return result


def draw_polyline(img: np.ndarray = None, dots: list = [[100, 260], [700, 260], [1100, 800], [0, 800]]) -> np.ndarray:
    pts = np.int32(dots)
    is_closed = True
    color = (255, 0, 0)  # Blue color in BGR
    thickness = 1  # Line thickness of 2 px
    # Using cv2.polylines() method draw a Blue polygon with thickness of 2 px
    result = cv2.polylines(img, [pts], is_closed, color, thickness)
    return result


@deprecated(version='1.0', reason='outdated')
def get_edges(img: np.ndarray = None) -> np.ndarray:
    # Перевести изображение в оттенки серого
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Найти границы изображения с помощью Canny edge detector
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    # Применить алгоритм преобразования Хафа
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)
    result = edges
    return result


def fix_rotated_page(img: np.ndarray = None) -> np.ndarray:
    # бинаризация изображения
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # морфологическая обработка
    kernel = np.ones((3, 3), np.uint8)
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    # поиск линий текста
    lines = cv2.HoughLinesP(closing, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)

    # оценка угла наклона текста
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.arctan2(y2 - y1, x2 - x1) * 180.0 / np.pi
        angles.append(angle)
    median_angle = np.median(angles)

    # поворот изображения
    rows, cols = img.shape[:2]
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), median_angle, 1)
    rotated = cv2.warpAffine(img, M, (cols, rows), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated


def fix_distorted_page(img: np.ndarray = None) -> np.ndarray:
    # бинаризация изображения
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # морфологическая обработка
    kernel = np.ones((3, 3), np.uint8)
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    # поиск линий текста
    lines = cv2.HoughLinesP(closing, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)

    # оценка угла наклона текста
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.arctan2(y2 - y1, x2 - x1) * 180.0 / np.pi
        angles.append(angle)
    median_angle = np.median(angles)

    # distort изображения
    # получение размеров изображения
    height, width = img.shape[:2]
    s = math.tan(math.radians(median_angle)) * width
    # print(s)

    # задание координат точек исходного изображения и соответствующих им точек искаженного изображения
    src_pts = np.float32([[0, 0], [width - 1, 0], [0, height - 1], [width - 1, height - 1]])
    dst_pts = np.float32([[0, 0], [width - 1, 0 - int(s )], [0, height - 1], [width - 1, height - 1 - int(s )]])
    # print(src_pts)
    # print(dst_pts)

    # получение матрицы преобразования
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    # применение преобразования
    distorted = cv2.warpPerspective(img, M, (width, height))
    return distorted


def fix_perspective_blob(img: np.ndarray = None) -> np.ndarray:
    # бинаризация изображения
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Настроить параметры детектора углов
    params = cv2.SimpleBlobDetector_Params()
    params.minThreshold = 10
    params.maxThreshold = 200
    params.filterByCircularity = False
    params.filterByConvexity = False
    params.filterByInertia = False
    params.filterByArea = True
    params.minArea = 10
    params.maxArea = 10000

    # Создать детектор углов и найти углы на изображении
    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(gray)

    # Определить четыре угла, которые окружают текст на изображении
    points = np.float32([kp.pt for kp in keypoints])
    rect = cv2.minAreaRect(points)
    box = cv2.boxPoints(rect)
    box = np.intp(box)

    # Вычислить матрицу перспективного преобразования для выравнивания изображения
    width, height = int(rect[1][0]), int(rect[1][1])
    print(width, height)
    # dst_points = np.array([[0, height - 1], [0, 0], [width - 1, 0], [width - 1, height - 1]], dtype="float32")
    s = 10 # сдвиг границ чтобы не обрезало края символов
    dst_points = np.array([[0 + s, height - 1 - s], [0 + s, 0 + s], [width - 1 - s, 0 + s], [width - 1 - s, height - 1 - s]], dtype="float32")
    M = cv2.getPerspectiveTransform(np.float32(box), dst_points)

    # Применить матрицу перспективного преобразования к изображению
    warped = cv2.warpPerspective(img, M, (width, height))
    return warped


def remove_non_text_areas_ocr(img: np.ndarray = None, blocks: int = 20, language: str = 'en') -> np.ndarray:
    # Преобразование изображения в оттенки серого
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Получение размеров изображения и определение размера блока и шаг перекрытия
    h, w = gray.shape
    block_size = int(h / blocks)
    overlap = int(block_size / 2)

    # Создание пустого массива для хранения вероятностей наличия символов
    probabilities = np.zeros((h, w))

    # Разбиение изображения на блоки с перекрытием и определение вероятности наличия символов в каждом блоке
    reader = easyocr.Reader([language], gpu=False)
    for i in range(0, h - block_size + 1, overlap):
        for j in range(0, w - block_size + 1, overlap):
            # Вырезать блок изображения
            block = gray[i:i + block_size, j:j + block_size]

            # Распознать текст с помощью EasyOCR
            result = reader.readtext(block)
            text = result[0][1] if result else ''

            # Определить вероятность наличия символов в блоке
            probability = 1 if text else 0

            # Заполнить соответствующую область в массиве вероятностей
            probabilities[i:i + block_size, j:j + block_size] += probability

    # Нормализовать массив вероятностей, чтобы значения были в диапазоне от 0 до 1
    # probabilities /= np.max(probabilities)
    probabilities /= ((block_size/overlap) ** 2)

    # Создать тепловую карту на основе массива вероятностей
    heatmap = cv2.applyColorMap((probabilities * 255).astype(np.uint8), cv2.COLORMAP_HOT)

    # # Сохранить тепловую карту в файл
    # cv2.imwrite('heatmap2.png', heatmap)

    # преобразование в градации серого и бинаризация второго изображения
    gray = cv2.cvtColor(heatmap, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)

    # создание нового белого изображения той же формы, что и исходное изображение
    new_img = np.zeros_like(img) + 255  # 255 - максимальное значение яркости для белого цвета

    # инвертирование маски
    inv_mask = cv2.bitwise_not(mask)

    # копирование только тех пикселей исходного изображения, которые соответствуют маске
    new_img[np.where(inv_mask == 0)] = img[np.where(inv_mask == 0)]

    # вместо предыдущих трех команд можно использовать
    # img[np.where(mask == 0)] = 255
    return new_img


def fix_perspective_dilate(img: np.ndarray = None) -> np.ndarray:
    # перспективное преобразование
    # бинаризация изображения
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=3)

    # нахождение контуров на бинаризованном изображении
    contours, hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # нахождение контура с максимальной площадью
    max_contour = max(contours, key=cv2.contourArea)
    # аппроксимация многоугольника с помощью алгоритма Дугласа-Пекера
    epsilon = 0.05 * cv2.arcLength(max_contour, True)
    approx = cv2.approxPolyDP(max_contour, epsilon, True)
    # построение Convex Hull для множества точек многоугольника
    hull = cv2.convexHull(approx)
    # рисование Convex Hull на изображении
    # cv2.drawContours(img, [hull], 0, (0, 255, 0), 2)
    # print(hull)

    # Преобразуем массив hull к одномерному массиву
    hull_flat = hull.ravel()
    # Разделим одномерный массив на подмассивы, каждый из которых содержит координаты одной точки
    points = np.split(hull_flat, hull.shape[0])
    # Отсортируем подмассивы по возрастанию координаты x
    points_sorted = sorted(points, key=lambda x: x[0])
    # Выберем первые два подмассива с отсортированными координатами и преобразуем их обратно к формату массива
    left_points = np.array(points_sorted[:2]).reshape((-1, 1, 2))
    # print(left_points)

    # координаты точек
    x1, y1 = left_points[0][0]
    x2, y2 = left_points[1][0]
    # расчет угла в градусах
    dx = x2 - x1
    dy = y2 - y1
    angle = 180 - (math.atan2(dx, dy) * 180 / math.pi)
    # приведение угла к диапазону [0, 360)
    angle = (angle + 360) % 360
    # print(angle)

    rows, cols = img.shape[:2]
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    rotated = cv2.warpAffine(img, M, (cols, rows), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    return rotated
