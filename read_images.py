import cv2
import matplotlib.pyplot as plt
import numpy as np

def resize_to_28x28(img):
    img_h, img_w = img.shape
    print(img.shape)
    dim_size_max = max(img.shape)

    if dim_size_max == img_w:
        im_h = (26 * img_h) // img_w
        if im_h <= 0 or img_w <= 0:
            print("Invalid Image Dimension: ", im_h, img_w, img_h)
        tmp_img = cv2.resize(img, (26,im_h),0,0,cv2.INTER_NEAREST)
    else:
        im_w = (26 * img_w) // img_h
        if im_w <= 0 or img_h <= 0:
            print("Invalid Image Dimension: ", im_w, img_w, img_h)
        tmp_img = cv2.resize(img, (im_w, 26),0,0,cv2.INTER_NEAREST)

    out_img = np.zeros((28, 28), dtype=np.ubyte)

    nb_h, nb_w = out_img.shape
    na_h, na_w = tmp_img.shape
    y_min = (nb_w) // 2 - (na_w // 2)
    y_max = y_min + na_w
    x_min = (nb_h) // 2 - (na_h // 2)
    x_max = x_min + na_h

    out_img[x_min:x_max, y_min:y_max] = 255 - tmp_img
    out_img[out_img < 130] = 0
    return out_img

path = r"PATH.jpg" #7
img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)

tsr_img = resize_to_28x28(img)
input_data = np.copy(tsr_img).reshape(1,28,28,1)
print(np.argmax(model.predict(input_data)))

plt.subplot(330 + 1)
image = input_data

plt.imshow((image[0]).astype('uint8'))
plt.show()
