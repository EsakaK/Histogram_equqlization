import cv2
import numpy as np
import matplotlib.pyplot as plt

class Histogram_equalization(object):
    def __init__(self,ImgFile):
        self.OriginalImg = cv2.imread(ImgFile)
        self.Histogram = np.zeros((256,3),dtype=np.int)
        self.NewHistogram = np.zeros((256,3),dtype=np.int)

    def Histogram_equalizing(self):
        # 1.计算像素点，得到直方图
        height = self.OriginalImg.shape[0]
        width = self.OriginalImg.shape[1]
        channels = self.OriginalImg.shape[2]
        for row in range(height):
            for col in range(width):
                for channel in range(channels):
                    I = self.OriginalImg[row][col][channel]
                    self.Histogram[I][channel]+=1
        # 绘制直方图
        x = list(range(256))
        plt.figure()
        plt.xlim(0,255)
        plt.plot(x,self.Histogram[:,0],color = 'red', linewidth = 0.5)
        plt.plot(x, self.Histogram[:, 1], color='green', linewidth=0.5)
        plt.plot(x, self.Histogram[:, 2], color='blue', linewidth=0.5)
        plt.savefig('Histogram.jpg')

        # 2.直方图出来了，接着计算累积分布函数
        # 使用c(I)的计算公式
        N = height * width # 所有像素点个数
        Cumulative_distribution = np.zeros((256,3),dtype=np.float)
        for i in range(256):
            for channel in range(channels):
                if(i == 0):
                    Cumulative_distribution[i][channel] = self.Histogram[i][channel]/N
                elif(i==255):
                    Cumulative_distribution[i][channel] = 1.0
                else:
                    Cumulative_distribution[i][channel] = self.Histogram[i][channel]/N + Cumulative_distribution[i-1][channel]

        # 绘制图像
        x = list(range(256))
        plt.figure()
        plt.xlim(0,255)
        plt.plot(x, Cumulative_distribution[:, 0], color='red', linewidth=0.5)
        plt.plot(x, Cumulative_distribution[:, 1], color='green', linewidth=0.5)
        plt.plot(x, Cumulative_distribution[:, 2], color='blue', linewidth=0.5)
        plt.savefig('Cumulative_distribution_NonStandardization.jpg')

        # 3.累积分布函数计算完成后，进行I和c的缩放，把值域缩放到0~255的范围之内
        self.Cumulative_distribution =Cumulative_distribution * 255
        # 绘制图像
        x = list(range(256))
        plt.figure()
        plt.xlim(0,255)
        plt.plot(x, self.Cumulative_distribution[:, 0], color='red', linewidth=0.5)
        plt.plot(x, self.Cumulative_distribution[:, 1], color='green', linewidth=0.5)
        plt.plot(x, self.Cumulative_distribution[:, 2], color='blue', linewidth=0.5)
        plt.savefig('Cumulative_distribution.jpg')

        # 4.对均衡后的图像进行平滑处理,使用线性混合
        f = np.zeros((256,3),dtype=np.int)
        alpha = 1
        for i in range(256):
            for channel in range(channels):
                f[i][channel] = alpha*self.Cumulative_distribution[i][channel] + (1-alpha)*i
        self.f = f.astype(np.int)

        # 5.f为得到的映射，据此生成新的图像
        self.NewImg = np.zeros((self.OriginalImg.shape))
        self.NewImg = self.OriginalImg.copy()
        for row in range(height):
            for col in range(width):
                for channel in range(channels):
                    Newvalue = f[self.NewImg[row][col][channel]][channel]
                    self.NewImg[row][col][channel] = Newvalue
        cv2.imwrite('NewTest.jpg',self.NewImg)
        # 统计新图像的直方图
        for row in range(height):
            for col in range(width):
                for channel in range(channels):
                    I = self.NewImg[row][col][channel]
                    self.NewHistogram[I][channel]+=1
        tmpM = np.transpose(self.NewHistogram)
        z1 = np.where(tmpM[0]==0)
        z2 = np.where(tmpM[1]==0)
        z3 = np.where(tmpM[2] == 0)
        y1 = np.delete(tmpM[0],z1)
        y2 = np.delete(tmpM[1], z2)
        y3 = np.delete(tmpM[2], z3)
        x1 = np.array(np.where(tmpM[0]!=0))[0]
        x2 = np.array(np.where(tmpM[1]!=0))[0]
        x3 = np.array(np.where(tmpM[2]!=0))[0]


        plt.figure()
        plt.xlim(0, 255)
        plt.plot(x1, y1, color='red', linewidth=0.5)
        plt.plot(x2, y2, color='green', linewidth=0.5)
        plt.plot(x3, y3, color='blue', linewidth=0.5)
        plt.savefig('New_Histogram.jpg')

if __name__ == '__main__':
    HE = Histogram_equalization('Test.jpg')
    HE.Histogram_equalizing()