import cv2
import numpy as np
from skimage import color
from scipy.ndimage import gaussian_filter

def saliency_detection(img):
    """
    计算图像的显著性图。

    参数:
    ----------
    img : numpy.ndarray
        输入的RGB图像。

    返回:
    -------
    sm : numpy.ndarray
        输出的显著性图。
    """
    # 1. 对图像进行高斯模糊
    gfrgb = gaussian_filter(img, sigma=3)

    # 2. 将图像从 RGB 转换为 Lab 颜色空间
    lab = color.rgb2lab(gfrgb)

    # 3. 计算 Lab 颜色空间的平均值
    l, a, b = cv2.split(lab)
    lm, am, bm = np.mean(l), np.mean(a), np.mean(b)

    # 4. 计算显著性图
    sm = (l - lm)**2 + (a - am)**2 + (b - bm)**2

    return sm

# 测试代码
if __name__ == "__main__":
    # 读取图像
    img = cv2.imread('input_image.jpg')
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 计算显著性图
    saliency_map = saliency_detection(img_rgb)

    # 显示显著性图
    cv2.imshow('Saliency Map', saliency_map / np.max(saliency_map))  # 归一化显示
    cv2.waitKey(0)
    cv2.destroyAllWindows()
