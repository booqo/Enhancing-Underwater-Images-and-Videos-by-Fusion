import cv2
import numpy as np
import simple_color_balance as scb
import rgb_to_lab as tolab
import bilateralFilter as filter
from skimage import exposure, color
import matplotlib.pyplot as plt
from scipy.ndimage import convolve
from scipy.signal import convolve2d
import sm
import fusion
import copy
import test_fusion as fusion


def main():
    """-----------------------------读图像----------------------------------"""
    # orange_image_path = "./image/273.jpg"
    orange_image_path = "./image/8.jpg"
    # orange_image_path = "../underwater_image_fusion-master/images/2.jpg"

    image = cv2.imread(orange_image_path)

    # 检查图像是否成功读取
    if image is None:
        print(f"Error: Unable to read image at {orange_image_path}")
        return

    """----------------------------图像处理-----------------------------------"""
    # 算法输入的图像1
    img1 = scb.simple_color_balance(image)

    # 输出bgr图像
    # 函数内置转换，输入bgr，输出rgb给rgb_to_lab使用
    lab1 = tolab.rgb_to_lab(img1)

    # 输入2
    lab2 = copy.deepcopy(lab1)

    # 对亮度通道（L通道）应用双边滤波

    lab2[:, :, 0] = np.uint8(filter.bilateral_filter(np.double(lab2[:, :, 0])))  # 手动实现
    # lab2[:, :, 0] = cv2.bilateralFilter(lab2[:, :, 0].astype(np.float32), d=6, sigmaColor=5, sigmaSpace=5) # 調庫
    # 自适应直方图均衡化
    # 对亮度通道进行自适应直方图均衡化
    lab2[:, :, 0] = exposure.equalize_adapthist(lab2[:, :, 0] / 100, clip_limit=0.03) * 100
    # 确保 L 通道在 0-100 范围内
    lab2[:, :, 0] = np.clip(lab2[:, :, 0], 0, 100)

    # 将 Lab 图像转换回 RGB
    img2 = color.lab2rgb(lab2) * 255
    img2 = np.clip(img2, 0, 255).astype(np.uint8)
    img2 = cv2.cvtColor(img2.astype('uint8'), cv2.COLOR_RGB2BGR)

    '''``````````````````````计算权重 `````````````````````````````'''

    R1 = np.round(lab1[:, :, 0] / 100.0)  # 将 L 通道归一化到 [0, 1] 范围
    R2 = np.round(lab2[:, :, 0] / 100.0)

    # 1. 拉普拉斯对比权重 (Laplacian contrast weight)
    laplacian_kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])/8
    WL1 = np.abs(convolve(R1, laplacian_kernel, mode='reflect'))
    WL2 = np.abs(convolve(R2, laplacian_kernel, mode='reflect'))

    # testWL1 = cv2.Laplacian(R1, cv2.CV_64F)
    # testWL2 = cv2.Laplacian(R2, cv2.CV_64F)

    # 2. 局部对比权重 (Local contrast weight)
    h = (1 / 16) * np.array([1, 4, 6, 4, 1])
    whc = np.pi / 2.75  # 高频阻断值

    WLC1 = convolve2d(R1, np.outer(h, h), mode='same', boundary='wrap')
    WLC1[WLC1 > whc] = whc
    WLC1 = (R1 - WLC1) ** 2

    WLC2 = convolve2d(R2, np.outer(h, h), mode='same', boundary='wrap')
    WLC2[WLC2 > whc] = whc
    WLC2 = (R2 - WLC2) ** 2

    # 3. 显著性权重 (Saliency weight)
    WS1 = sm.saliency_detection(img1)  # 你需要实现的显著性检测函数(已实现)
    WS2 = sm.saliency_detection(img2)  # 你需要实现的显著性检测函数

    # 4. 曝光权重 (Exposedness weight)
    average = 0.5
    sigma = 0.25

    WE1 = np.exp(-((R1 - average) ** 2) / (2 * sigma ** 2))
    WE2 = np.exp(-((R2 - average) ** 2) / (2 * sigma ** 2))

    # 归一化权重 (Normalized weight)
    denominator = WL1 + WLC1 + WS1 + WE1 + WL2 + WLC2 + WS2 + WE2
    W1 = (WL1 + WLC1 + WS1 + WE1) / denominator
    W2 = (WL2 + WLC2 + WS2 + WE2) / denominator

    # W1 = np.ones(img1.shape[:2]) * 0.2
    # W2 = np.ones(img2.shape[:2]) * 0.8

    '''
    图像融合
    R(x,y) = sum G{W} * L{I}
    '''

    fusion_image = fusion.fuseTwoImages(img1, img2, W1, W2, 5)

    # cv2.imshow('Orange_image', Orange_image_rgb)
    # cv2.imshow('Image1', img1)
    # cv2.imshow('Image2', img2)
    #
    # cv2.imshow('Fusion Image', fusion_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # 使用 matplotlib 显示图像
    # 原图是由opencv读取的，默认BGR，使用cv2.cvtColor函数将 BGR 转换为 RGB给plt显示
    plt.subplot(2, 2, 1)  # 第一个子图
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Orange image")

    plt.subplot(2, 2, 2)
    plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    plt.title("imput image1")

    plt.subplot(2, 2, 3)
    plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    plt.title("imput image2")

    plt.subplot(2, 2, 4)
    plt.imshow(cv2.cvtColor(fusion_image, cv2.COLOR_BGR2RGB))
    plt.title("fusion image")
    plt.show()

if __name__ == '__main__':
    main()
