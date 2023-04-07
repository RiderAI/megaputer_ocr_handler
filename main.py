"""
A project to solve the problem of correcting distortions on scanned pages to improve recognition by means of OCR.
"""
import image_file_handler as ifh
import image_transform_handler as ith


def main():
    # ifh.show(ifh.load('image.png'), 'image.png')
    img = ifh.load('page_clean_mini_rotated.jpg')
    # img1 = ith.perspective_correction(img)
    # img2 = ith.draw_polyline(img)
    # ifh.show(img1)
    # ith.draw_polyline(img)
    # ifh.show(img, ith.get_edges(img))
    ifh.show_image(ith.fix_distorted_page(ith.fix_perspective_dilate(img)))

if __name__ == '__main__':
    main()
