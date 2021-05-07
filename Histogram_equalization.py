import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib


class HistogramEqualization(object):
    def __init__(self, img_file):
        self.OriginalImg = cv2.imread(img_file)
        self.Histogram = np.zeros((1, 256), dtype=np.int)
        self.NewHistogram = np.zeros((1, 256), dtype=np.int)
        self.Cumulative_distribution = np.zeros((1, 256), dtype=np.float)
        matplotlib.rcParams['font.sans-serif'] = ['SimHei']
        matplotlib.rcParams['axes.unicode_minus'] = False

    def resize(self, size: tuple = (64, 64)):
        """
        调用则会resize原始图片大小
        :return: None
        """
        self.OriginalImg = cv2.resize(self.OriginalImg, size)

    def draw_histogram(self):
        """
        转为灰度图，统计像素灰度数据，绘制直方图
        :return: None
        """
        # 先转化为灰度图
        self.OriginalImg = cv2.cvtColor(self.OriginalImg, cv2.COLOR_BGR2GRAY)
        cv2.imwrite('./result/origin_img.jpg', self.OriginalImg)
        # 计算像素点，得到原始图片直方图
        height = self.OriginalImg.shape[0]
        width = self.OriginalImg.shape[1]
        for row in range(height):
            for col in range(width):
                I = self.OriginalImg[row][col]
                self.Histogram[0][I] += 1
        # 绘制直方图
        x = np.asarray(self.OriginalImg)
        x.resize((height * width))
        plt.figure()
        plt.hist(x, bins=256, color='green')
        plt.xlabel('灰度值')
        plt.ylabel('频数')
        plt.title('原始图片直方图')
        plt.savefig('./result/origin_histogram.jpg')
        x.resize((height, width))

    def draw_cd_f(self):
        """
        累加直方图，得到累计分布函数，并绘制图像
        :return: None
        """
        # 使用c(I)的计算公式
        height = self.OriginalImg.shape[0]
        width = self.OriginalImg.shape[1]
        N = height * width  # 所有像素点个数
        for i in range(256):
            if i == 0:
                self.Cumulative_distribution[0][i] = self.Histogram[0][i] / N
            elif i == 255:
                self.Cumulative_distribution[0][i] = 1.0
            else:
                self.Cumulative_distribution[0][i] = self.Histogram[0][i] / N + \
                                                     self.Cumulative_distribution[0][i - 1]
        # 绘制图像
        x = list(range(256))
        plt.figure()
        plt.xlim(0, 255)
        plt.ylim(0, 1)
        plt.xlabel('灰度值')
        plt.ylabel('概率')
        plt.title('累积分布函数图像')
        plt.plot(x, self.Cumulative_distribution[0, :], color='green', linewidth=0.5)
        plt.savefig('./result/cdf.jpg')

    def equalization(self):
        """
        利用计算得到的累积分布函数对原始图像像素进行均衡化，得到映射函数
        :return: None
        """
        # 累积分布函数计算完成后，进行I和c的缩放，把值域缩放到0~255的范围之内
        self.Cumulative_distribution = self.Cumulative_distribution * 255
        height = self.OriginalImg.shape[0]
        width = self.OriginalImg.shape[1]
        # 对均衡后的图像进行平滑处理,使用线性混合
        f = np.zeros((1, 256), dtype=np.int)  # 映射函数
        alpha = 1  # 混合参数
        for i in range(256):
            f[0][i] = alpha * self.Cumulative_distribution[0][i] + (1 - alpha) * i
        self.f = f.astype(np.int)

        # f为得到的映射，据此生成新的图像
        self.NewImg = np.zeros((self.OriginalImg.shape))
        self.NewImg = self.OriginalImg.copy()
        for row in range(height):
            for col in range(width):
                Newvalue = f[0][self.NewImg[row][col]]
                self.NewImg[row][col] = Newvalue
        cv2.imwrite('./result/new_img.jpg', self.NewImg)

    def draw_new_histogram(self,filename='./result/new_histogram.jpg'):
        """绘制新图片的直方图"""
        # 计算像素点，得到原始图片直方图
        height = self.OriginalImg.shape[0]
        width = self.OriginalImg.shape[1]
        for row in range(height):
            for col in range(width):
                I = self.NewImg[row][col]
                self.NewHistogram[0][I] += 1
        # 绘制直方图
        self.NewImg.resize((height*width))
        plt.figure()
        plt.hist(self.NewImg, bins=256, color='green')
        plt.xlabel('灰度值')
        plt.ylabel('频数')
        plt.title('强化后图片直方图')
        plt.savefig(filename)
        self.NewImg.resize((height, width))

