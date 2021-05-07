from Histogram_equalization import HistogramEqualization

if __name__ == '__main__':
    HE = HistogramEqualization('./pic/Test.jpg')  # 建立实例对象
    HE.resize((224, 224))  # 归一化图像尺寸
    HE.draw_histogram()  # 画原始图像直方图
    HE.draw_cd_f()  # 画原始图像累积分布函数
    HE.equalization() # 利用以上计算得到的累积分布函数对图像进行均衡化，得到映射函数f,并使用映射函数f对原始图像进行均衡化，得到新的图像
    HE.draw_new_histogram() # 画新图像的直方图

