import cv2
import urllib.request
import numpy
from datetime import datetime


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


def show(img: numpy.ndarray = None, message: str = 'Image'):
    '''
    Shows an image in the GUI and disappears after pressing any key
    :param img: numpy.ndarray
    :param message:
    '''
    cv2.imshow(message, img) # Displaying the image Using cv2.imshow() method
    cv2.waitKey(0)           # waits for user to press any key (this is necessary to avoid Python kernel form crashing)
    cv2.destroyAllWindows()  # closing all open windows


def save(img: numpy.ndarray = None, path: str = None):
    if path is None:
        path = str(datetime.now()).replace(':', '-') + '.png'
        print(datetime.now())
    elif not path.endswith(('.png', '.jpg')):
        path += '.png'
    cv2.imwrite(path, img)

def main():
    # show(load('image.png'), 'image.png')
    save(load('image.png'))

if __name__ == '__main__':
    main()