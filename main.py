"""
A project to solve the problem of correcting distortions on scanned pages to improve recognition by means of OCR.
Проект по решению проблемы исправления искажений на отсканированных страницах для улучшения распознавания с помощью OCR.
"""
import image_file_handler as ifh
import image_transform_handler as ith


def main():
    img = ifh.load('page_mini.jpg')
    img0 = ith.remove_non_text_areas_ocr(img)
    img1 = ith.fix_perspective_dilate(img0)
    img2 = ith.fix_distorted_page(img1)
    print(ith.recognize_ocr(img2))
    ifh.show_image(img2)


if __name__ == '__main__':
    main()
