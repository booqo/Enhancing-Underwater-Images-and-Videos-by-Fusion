import cv2
import numpy as np
from skimage import color


# 将RGB转换为Lab颜色空间
def rgb_to_lab(rgbImage):
    # 检查图像是否为 BGR 格式，如果是，转换为 RGB
    if rgbImage.shape[2] == 3:  # 确保图像有 3 个通道
        rgbImage = cv2.cvtColor(rgbImage, cv2.COLOR_BGR2RGB)

    # 将RGB图像归一化到 [0, 1] 范围
    rgb_normalized = rgbImage / 255.0

    # 使用归一化后的图像进行Lab转换
    lab = color.rgb2lab(rgb_normalized)
    # lab[:, :, 0] = lab[:, :, 0] * 2.55
    # lab[:, :, 1] = lab[:, :, 1] + 128
    # lab[:, :, 2] = lab[:, :, 2] + 128
    # lab = np.uint8(lab)

    return lab


if __name__ == '__main__':
    # 读取图像
    image = cv2.imread('image/273.jpg')

    # 检查图像是否成功读取
    if image is None:
        print("Error: Unable to read image.")
    else:
        # 转换为 Lab 颜色空间
        lab_image = rgb_to_lab(image)

        # 如果你需要显示结果，必须先转换回 RGB 或 BGR 颜色空间
        display_image = color.lab2rgb(lab_image)
        display_image = (display_image * 255).astype(np.uint8)  # 转换为 8 位图像

        # 转换为 BGR 格式以使用 OpenCV 显示
        display_image_bgr = cv2.cvtColor(display_image, cv2.COLOR_RGB2BGR)

        # 显示结果
        cv2.imshow('Output Image', display_image_bgr)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
