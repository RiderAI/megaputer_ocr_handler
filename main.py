'''
A project to solve the problem of correcting distortions on scanned pages to improve recognition by means of OCR.
'''
import image_file_handler as ifh


def main():
    ifh.show(ifh.load('image.png'), 'image.png')


if __name__ == '__main__':
    main()
