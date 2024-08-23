import numpy as np
import cv2


def simple_color_balance(image):
    # 输入的图像是opencv读取的默认bgr
    # 将图像转换为浮点型进行处理
    image = image.astype(np.float64)

    # 注意penCV 加载的图像是 BGR 顺序

    # 分离RGB通道
    b, g, r = image[:, :, 0], image[:, :, 1], image[:, :, 2]

    # 计算每个通道的平均值

    Bavg = np.mean(b)
    Gavg = np.mean(g)
    Ravg = np.mean(r)

    # 计算最大平均值和每个通道的比例
    Max = max(Bavg, Gavg, Ravg)
    ratio = [Max / Bavg, Max / Gavg, Max / Ravg]

    # 设置饱和度水平
    satLevel = 0.005 * np.array(ratio)

    # 图像形状
    m, n, p = image.shape

    # 将每个通道展开为一维向量
    imgRGB_orig = np.zeros((p, m * n))
    imgRGB_orig[0, :] = b.flatten()
    imgRGB_orig[1, :] = g.flatten()
    imgRGB_orig[2, :] = r.flatten()

    # 直方图对比度调整
    imRGB = np.zeros_like(imgRGB_orig)

    for ch in range(p):
        q_low = satLevel[ch]
        q_high = 1 - satLevel[ch]

        # 计算quantile值
        tiles = np.quantile(imgRGB_orig[ch, :], [q_low, q_high])
        temp = imgRGB_orig[ch, :].copy()

        # 裁剪极值
        temp[temp < tiles[0]] = tiles[0]
        temp[temp > tiles[1]] = tiles[1]

        # 线性拉伸
        pmin = np.min(temp)
        pmax = np.max(temp)
        imRGB[ch, :] = (temp - pmin) * 255 / (pmax - pmin)

    # 将调整后的数据重新组合为图像
    output = np.zeros_like(image)
    output[:, :, 0] = imRGB[0, :].reshape((m, n))
    output[:, :, 1] = imRGB[1, :].reshape((m, n))
    output[:, :, 2] = imRGB[2, :].reshape((m, n))

    # 确保输出的类型为uint8
    output = output.astype(np.uint8)

    return output



# 测试代码
# 读取图像
if __name__ == '__main__':
    image = cv2.imread('image/273.jpg')
    # 读取gbr图像

    # 应用白平衡和对比度调整
    output = simple_color_balance(image)

    # 保存或显示结果
    cv2.imshow('output_image.jpg', output)
    cv2.waitKey(0)
