import cv2
import urllib.request
import numpy


def load(path: str = '') -> numpy.ndarray:
    '''
    Load image:numpy.ndarray from url or local file
    :param path: url or local file 'https://ds-blobs-3.cdn.devapps.ru/11999470.png', 'g4g.png'
    :raises fileNotFoundException
    :returns image:numpy.ndarray or None if file not found
    '''

    try:
        if path.lower().startswith(('http://', 'https://')):
            req = urllib.request.urlopen(path)
            arr = numpy.asarray(bytearray(req.read()), dtype=numpy.uint8)
            img = cv2.imdecode(arr, -1)
        else:
            img = cv2.imread(path)
    except:
        img = None
    return img


def show(img: numpy.ndarray = None, message: str = 'image'):
    cv2.imshow(message, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    show(load('image.png'), 'image.png')


if __name__ == '__main__':
    main()

# Save image in set directory
# Read RGB image
# img = cv2.imread('g4g.png')

# print(type(img))
# Output img with window name as 'image'
# cv2.imshow('image', img)
# Maintain output window utill user presses a key
# cv2.waitKey(0)
# Destroying present windows on screen
# cv2.destroyAllWindows()

# req = urllib.request.urlopen('https://ds-blobs-3.cdn.devapps.ru/11999470.png')
# arr = numpy.asarray(bytearray(req.read()), dtype=numpy.uint8)
# img = cv2.imdecode(arr, -1) # 'Load it as it is'
#
# cv2.imshow('lalala', img)
# if cv2.waitKey() & 0xff == 27: quit()
