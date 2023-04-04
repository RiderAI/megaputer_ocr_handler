import cv2
import urllib.request
import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from datetime import datetime
from deprecated import deprecated


def load(path: str = '') -> np.ndarray:
    """
    Load image:numpy.ndarray from url or local file
    :param path: url or local file 'https://ds-blobs-3.cdn.devapps.ru/11999470.png', 'g4g.png'
    :raises fileNotFoundException
    :returns image:numpy.ndarray or None if file not found
    """
    try:
        if path.lower().startswith(('http://', 'https://')):
            req = urllib.request.urlopen(path)
            arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
            img = cv2.imdecode(arr, -1)
        else:
            img = cv2.imread(path)
    except FileNotFoundError:
        print("Error: file not found")
    except Exception as e:
        print(f"Error: {e}")
    return img


@deprecated(version='1.0', reason='Replaced by show()')
def show_image(img: np.ndarray = None, message: str = 'Image'):
    """
    Shows an image in the GUI and disappears after pressing any key
    :param img: numpy.ndarray
    :param message:
    """
    cv2.imshow(message, img)  # Displaying the image Using cv2.imshow() method
    cv2.waitKey(0)  # waits for user to press any key (this is necessary to avoid Python kernel form crashing)
    cv2.destroyAllWindows()  # closing all open windows


def show(*imgs: np.ndarray):
    n_plots = len(imgs)
    n_rows = int(np.ceil(np.sqrt(n_plots)))
    n_lines = int(np.ceil(n_plots / n_rows))
    for i, img in enumerate(imgs):
        plt.subplot(n_rows, n_lines, i + 1), plt.imshow(img), plt.title(f'Image {i + 1}')
    plt.show()


def save(img: np.ndarray = None, path: str = None):
    if path is None:
        path = str(datetime.now()).replace(':', '-') + '.png'
    elif not path.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
        path += '.png'
    cv2.imwrite(path, img)


def main():
    # show(load('image.png'), 'image.png')
    save(load('image.png'), 'out_image.png')


if __name__ == '__main__':
    main()
