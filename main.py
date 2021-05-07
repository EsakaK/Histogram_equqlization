from Histogram_equalization import HistogramEqualization

if __name__ == '__main__':
    HE = HistogramEqualization('./pic/Test.jpg')  # 建立实例对象
    HE.resize((224, 224))  # 归一化图像尺寸
    HE.draw_histogram()  # 画原始图像直方图
    HE.draw_cd_f()  # 画原始图像累积分布函数
