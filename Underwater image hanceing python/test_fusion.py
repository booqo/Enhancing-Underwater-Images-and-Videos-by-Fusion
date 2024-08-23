import cv2
import numpy as np


# 创建高斯滤波器掩码
def filterMask(img):
    """
    创建高斯滤波器掩码，用于生成高斯金字塔。

    参数:
    - img: 输入图像，用于确定数据类型。

    返回:
    - mask: 生成的高斯滤波掩码。
    """
    h = [1.0 / 16.0, 4.0 / 16.0, 6.0 / 16.0, 4.0 / 16.0, 1.0 / 16.0]
    mask = np.zeros((len(h), len(h)), dtype=img.dtype)
    for i in range(len(h)):
        for j in range(len(h)):
            mask[i, j] = h[i] * h[j]
    return mask


# 构建高斯金字塔
def build_gaussian_pyramid(img, levels):
    """
    构建高斯金字塔，用于多分辨率图像处理。

    参数:
    - img: 输入图像。
    - levels: 金字塔层数。

    返回:
    - gaussPyr: 高斯金字塔的每一层图像列表。
    """
    gaussPyr = [img.copy()]  # 第一层为原始图像
    current_img = img.copy()

    for i in range(1, levels):
        # 先进行下采样，再进行高斯滤波
        current_img = cv2.pyrDown(current_img)  # 使用 pyrDown 进行降采样
        gaussPyr.append(current_img.copy())  # 将当前图像添加到金字塔中

    return gaussPyr


# 构建拉普拉斯金字塔
def build_laplacian_pyramid(img, levels):
    """
    构建拉普拉斯金字塔，通过高斯金字塔的差异生成。

    参数:
    - img: 输入图像。
    - levels: 金字塔的层数。

    返回:
    - lapPyr: 包含拉普拉斯金字塔各层的列表。
    """
    gaussPyr = build_gaussian_pyramid(img, levels)
    lapPyr = []

    for i in range(levels - 1):
        # 计算当前层与下层之间的差值，得到拉普拉斯层
        size = (gaussPyr[i].shape[1], gaussPyr[i].shape[0])
        upsampled = cv2.pyrUp(gaussPyr[i + 1], dstsize=size)
        laplacian = cv2.subtract(gaussPyr[i], upsampled)
        lapPyr.append(laplacian)

    lapPyr.append(gaussPyr[-1])  # 最顶层为高斯金字塔的最高层
    return lapPyr


# 拉普拉斯金字塔重建图像
def reconstructLaplacianPyramid(pyramid):
    """
    从拉普拉斯金字塔重建图像。

    参数:
    - pyramid: 拉普拉斯金字塔的各层图像列表。

    返回:
    - 重建后的图像。
    """
    reconstructed_img = pyramid[-1]  # 从最顶层开始重建
    for i in range(len(pyramid) - 2, -1, -1):
        upsampled = cv2.pyrUp(reconstructed_img, dstsize=(pyramid[i].shape[1], pyramid[i].shape[0]))
        reconstructed_img = cv2.add(pyramid[i], upsampled)
    return reconstructed_img


# 融合两幅图像
def fuseTwoImages(img1, img2, w1, w2, level):
    """
    使用多分辨率方法融合两幅图像。

    参数:
    - w1: 第一幅图像的权重图。
    - img1: 第一幅图像。
    - w2: 第二幅图像的权重图。
    - img2: 第二幅图像。
    - level: 金字塔的层数。

    返回:
    - 融合后的图像。
    """
    # 构建权重的高斯金字塔
    weight1 = build_gaussian_pyramid(w1.astype(np.float32), level)
    weight2 = build_gaussian_pyramid(w2.astype(np.float32), level)

    # 将图像转换为浮点类型
    img1 = np.float32(img1)
    img2 = np.float32(img2)

    # 分离 RGB 通道
    bCnl1, gCnl1, rCnl1 = cv2.split(img1)
    bCnl2, gCnl2, rCnl2 = cv2.split(img2)

    # 构建各通道的拉普拉斯金字塔
    bPyr1 = build_laplacian_pyramid(bCnl1, level)
    gPyr1 = build_laplacian_pyramid(gCnl1, level)
    rPyr1 = build_laplacian_pyramid(rCnl1, level)

    bPyr2 = build_laplacian_pyramid(bCnl2, level)
    gPyr2 = build_laplacian_pyramid(gCnl2, level)
    rPyr2 = build_laplacian_pyramid(rCnl2, level)

    # 对各层的拉普拉斯金字塔进行融合
    fused_bPyr = [
        cv2.add(
            cv2.multiply(bPyr1[i], weight1[i], dtype=cv2.CV_32F),
            cv2.multiply(bPyr2[i], weight2[i], dtype=cv2.CV_32F)
        ) for i in range(level)
    ]
    fused_gPyr = [
        cv2.add(
            cv2.multiply(gPyr1[i], weight1[i], dtype=cv2.CV_32F),
            cv2.multiply(gPyr2[i], weight2[i], dtype=cv2.CV_32F)
        ) for i in range(level)
    ]
    fused_rPyr = [
        cv2.add(
            cv2.multiply(rPyr1[i], weight1[i], dtype=cv2.CV_32F),
            cv2.multiply(rPyr2[i], weight2[i], dtype=cv2.CV_32F)
        ) for i in range(level)
    ]

    # 重建融合后的图像通道
    bChannel = reconstructLaplacianPyramid(fused_bPyr)
    gChannel = reconstructLaplacianPyramid(fused_gPyr)
    rChannel = reconstructLaplacianPyramid(fused_rPyr)

    # 合并通道
    fusion = cv2.merge((bChannel, gChannel, rChannel))

    # 线性映射到 [0, 255] 范围
    min_val, max_val = fusion.min(), fusion.max()
    fusion = 255 * (fusion - min_val) / (max_val - min_val)
    return fusion.astype(np.uint8)

    # # 将结果裁剪到 0-255 范围并转换为 uint8 类型
    # return np.clip(fusion, 0, 255).astype(np.uint8)

# 主程序：图像融合的示例调用
if __name__ == "__main__":
    # 读取两个待融合的图像
    img1 = cv2.imread('../balanced_image.jpg')
    img2 = cv2.imread('../balanced_image.jpg')

    # 计算权重（这里假设 W1 和 W2 已经计算好，简单设置为全 1 的权重图）
    W1 = np.ones(img1.shape[:2])
    W2 = np.ones(img2.shape[:2])

    # 执行图像融合
    fused_image = fuseTwoImages(img1, img2, W1, W2, 5)

    # 显示融合后的图像结果
    cv2.imshow('Fused Image', fused_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

