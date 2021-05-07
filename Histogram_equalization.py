import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib


class HistogramEqualization(object):
    def __init__(self, img_file):
        self.OriginalImg = cv2.imread(img_file)
        self.Histogram = np.zeros((1, 256), dtype=np.int)
        self.NewHistogram = np.zeros((1, 256), dtype=np.int)
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
        Cumulative_distribution = np.zeros((1, 256), dtype=np.float)
        for i in range(256):
            if i == 0:
                Cumulative_distribution[0][i] = self.Histogram[0][i] / N
            elif i == 255:
                Cumulative_distribution[0][i] = 1.0
            else:
                Cumulative_distribution[0][i] = self.Histogram[0][i] / N + \
                                                Cumulative_distribution[0][i - 1]
        # 绘制图像
        x = list(range(256))
        plt.xlim(0, 255)
        plt.xlabel('灰度值')
        plt.ylabel('概率')
        plt.title('累积分布函数图像')
        plt.plot(x, Cumulative_distribution[0, :], color='green', linewidth=0.5)
        plt.savefig('./result/cdf.jpg')

    def Histogram_equalizing(self):
        # 1.计算像素点，得到直方图
        height = self.OriginalImg.shape[0]
        width = self.OriginalImg.shape[1]
        channels = self.OriginalImg.shape[2]
        for row in range(height):
            for col in range(width):
                for channel in range(channels):
                    I = self.OriginalImg[row][col][channel]
                    self.Histogram[I][channel] += 1
        # 绘制直方图
        x = list(range(256))
        plt.figure()
        plt.xlim(0, 255)
        plt.plot(x, self.Histogram[:, 0], color='red', linewidth=0.5)
        plt.plot(x, self.Histogram[:, 1], color='green', linewidth=0.5)
        plt.plot(x, self.Histogram[:, 2], color='blue', linewidth=0.5)
        plt.savefig('Histogram.jpg')

        # 2.直方图出来了，接着计算累积分布函数
        # 使用c(I)的计算公式
        N = height * width  # 所有像素点个数
        Cumulative_distribution = np.zeros((256, 3), dtype=np.float)
        for i in range(256):
            for channel in range(channels):
                if (i == 0):
                    Cumulative_distribution[i][channel] = self.Histogram[i][channel] / N
                elif (i == 255):
                    Cumulative_distribution[i][channel] = 1.0
                else:
                    Cumulative_distribution[i][channel] = self.Histogram[i][channel] / N + \
                                                          Cumulative_distribution[i - 1][channel]

        # 绘制图像
        x = list(range(256))
        plt.figure()
        plt.xlim(0, 255)
        plt.plot(x, Cumulative_distribution[:, 0], color='red', linewidth=0.5)
        plt.plot(x, Cumulative_distribution[:, 1], color='green', linewidth=0.5)
        plt.plot(x, Cumulative_distribution[:, 2], color='blue', linewidth=0.5)
        plt.savefig('Cumulative_distribution_NonStandardization.jpg')

        # 3.累积分布函数计算完成后，进行I和c的缩放，把值域缩放到0~255的范围之内
        self.Cumulative_distribution = Cumulative_distribution * 255
        # 绘制图像
        x = list(range(256))
        plt.figure()
        plt.xlim(0, 255)
        plt.plot(x, self.Cumulative_distribution[:, 0], color='red', linewidth=0.5)
        plt.plot(x, self.Cumulative_distribution[:, 1], color='green', linewidth=0.5)
        plt.plot(x, self.Cumulative_distribution[:, 2], color='blue', linewidth=0.5)
        plt.savefig('Cumulative_distribution.jpg')

        # 4.对均衡后的图像进行平滑处理,使用线性混合
        f = np.zeros((256, 3), dtype=np.int)
        alpha = 1
        for i in range(256):
            for channel in range(channels):
                f[i][channel] = alpha * self.Cumulative_distribution[i][channel] + (1 - alpha) * i
        self.f = f.astype(np.int)

        # 5.f为得到的映射，据此生成新的图像
        self.NewImg = np.zeros((self.OriginalImg.shape))
        self.NewImg = self.OriginalImg.copy()
        for row in range(height):
            for col in range(width):
                for channel in range(channels):
                    Newvalue = f[self.NewImg[row][col][channel]][channel]
                    self.NewImg[row][col][channel] = Newvalue
        cv2.imwrite('NewTest.jpg', self.NewImg)
        # 统计新图像的直方图
        for row in range(height):
            for col in range(width):
                for channel in range(channels):
                    I = self.NewImg[row][col][channel]
                    self.NewHistogram[I][channel] += 1
        tmpM = np.transpose(self.NewHistogram)
        z1 = np.where(tmpM[0] == 0)
        z2 = np.where(tmpM[1] == 0)
        z3 = np.where(tmpM[2] == 0)
        y1 = np.delete(tmpM[0], z1)
        y2 = np.delete(tmpM[1], z2)
        y3 = np.delete(tmpM[2], z3)
        x1 = np.array(np.where(tmpM[0] != 0))[0]
        x2 = np.array(np.where(tmpM[1] != 0))[0]
        x3 = np.array(np.where(tmpM[2] != 0))[0]

        plt.figure()
        plt.xlim(0, 255)
        plt.plot(x1, y1, color='red', linewidth=0.5)
        plt.plot(x2, y2, color='green', linewidth=0.5)
        plt.plot(x3, y3, color='blue', linewidth=0.5)
        plt.savefig('New_Histogram.jpg')


if __name__ == '__main__':
    HE = Histogram_equalization('Test.jpg')
    HE.Histogram_equalizing()
