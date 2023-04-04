"""
A project to solve the problem of correcting distortions on scanned pages to improve recognition by means of OCR.
"""
import image_file_handler as ifh
import image_transform_handler as ith


def main():
    # ifh.show(ifh.load('image.png'), 'image.png')
    img = ifh.load('page.jpg')
    img1 = ith.perspective_correction(img)
    # ifh.show(img1)
    # ith.draw_polyline(img)
    ifh.show(ith.draw_polyline(img), img1, ith.get_edges(img))

if __name__ == '__main__':
    main()
