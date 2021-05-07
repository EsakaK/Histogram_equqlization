# Histogram equalization
直方图均衡化的python实现（结果展示）
## 应用背景
- 许多图片对比度不够，导致视觉效果不佳（不够清晰、难以分辨不同部分）。   
- 人类视觉系统都难以分辨的图片，再要求计算机去识别过于苛刻
<img src="https://github.com/EsakaKyo/Histogram_equqlization/blob/master/pic/Test.jpg" align="left" alt="test1" title="GitHub,Social Coding" width="200" height="200" />
<img src="https://github.com/EsakaKyo/Histogram_equqlization/blob/master/pic/Test(2).jpg" align="mid" alt="test2" title="GitHub,Social Coding" width="200" height="200" />

之所以会出现“看不清，难以分辨……”这些问题，和图片的很多属性有关。   
比如说分辨率、锐度、对比度等等。而对比度对图像视觉效果影响极为关键。   
高对比度的图像会十分醒目，色彩鲜艳；低对比度的图像则会十分模糊，显得灰蒙蒙。

<br/>   

## 基础概念
在了解如何使用直方图均衡化的方法提高图片清晰度之前，需要了解一些基础概念
### 1. 对比度(Contrast)   
对比度是一个反应图像亮度差异的值，对于人的视觉系统来说，如果视网膜接受的光的亮度之间差异较大，人会感到色彩绚丽，图像清晰。
        通常计算公式为    
        <img src="http://chart.googleapis.com/chart?cht=tx&chl=C=\frac{I_{max}+I_{v}}{I_{min}+I_{v}}" style="border:none;">

### 2. 直方图(Histogram)   
直方图是一个统计概念，用于统计一个图像中各个强度的像素点出现的个数。从而可以得到一个像素点的强度的概率分布。
图像 x 中单个像素点出现灰度 I 的概率为：   
        <img src="http://chart.googleapis.com/chart?cht=tx&chl= p_x(i)=\frac{n_i}{n}" style="border:none;">   
这其实是将0~255的灰度归一化到[0,1]。   
<img src="https://github.com/EsakaKyo/Histogram_equqlization/blob/master/result/origin_img.jpg" align="left" alt="test1" title="GitHub,Social Coding" width="200" height="200" />
<img src="https://github.com/EsakaKyo/Histogram_equqlization/blob/master/result/origin_histogram.jpg"  alt="test2" title="GitHub,Social Coding" width="200" height="200" />    
(左边为原始图片，右边为它的直方图)

### 3. 均衡化(Equalization)   
直方图的均衡化目的是使直方图尽可能平坦，从而避免整幅图像的强度集中在某一个很小的区域之内而导致分辨不清。   
为了使灰度尽可能平坦分布，需要构造一个映射c(I)，该映射可以考虑到整幅图像的强度变化，从而使灰度分布尽可能广泛。   
对于全局均衡化，可以采用累计分布函数(cumulative distribution function)作为该映射，因为累计分布函数会综合像素灰度值 I 之前所有灰度值出现的情况。   
累计分布函数的计算如下：   
<img src="http://chart.googleapis.com/chart?cht=tx&chl= c(I)=\frac{1}{N}\sum_{i=0}^{I}h(i)=c(I-1)+\frac{1}{N}h(I)" style="border:none;">    
得到的分布函数图：   
<img src="https://github.com/EsakaKyo/Histogram_equqlization/blob/master/pic/origin_cdf.jpg"  alt="test2" title="GitHub,Social Coding" width="200" height="200" />    


### 4. 局部补偿(Partial compensation)   
局部补偿可以改善得到的直方图，使其在保留较多原始灰度图像分布的同时达到更有吸引力的平衡。譬如如下简单的线性混合：   
<img src="http://chart.googleapis.com/chart?cht=tx&chl= f(I)=(1-\alpha)I + \alpha c(I)" style="border:none;">     
    
<br/>

## 代码展示
以下类中定义了所有方法，并且有详细操作注释，详见代码文件
```python
class Histogram_equalization(object):
    def __init__(self,ImgFile):
        self.OriginalImg = cv2.imread(ImgFile)
        self.Histogram = np.zeros((256,3),dtype=np.int)
        self.NewHistogram = np.zeros((256,3),dtype=np.int)
```

<br/>

## 实验结果
<img src="https://github.com/EsakaKyo/Histogram_equlization/blob/master/pic/NewTest(1).jpg"  align ="left" alt="test2" title="GitHub,Social Coding" width="200" height="200" />
<img src="https://github.com/EsakaKyo/Histogram_equlization/blob/master/pic/Test.jpg"  alt="test2" title="GitHub,Social Coding" width="200" height="200" />
   
<img src="https://github.com/EsakaKyo/Histogram_equlization/blob/master/pict/NewTest(2).jpg"  align ="left"  alt="test2" title="GitHub,Social Coding" width="200" height="200" />
<img src="https://github.com/EsakaKyo/Histogram_equlization/blob/master/pic/Test(2).jpg"  alt="test2" title="GitHub,Social Coding" width="200" height="200" />
   
<img src="https://github.com/EsakaKyo/Histogram_equlization/blob/master/pic/NewTest(3).jpg"  align ="left"  alt="test2" title="GitHub,Social Coding" width="200" height="200" /> 
<img src="https://github.com/EsakaKyo/Histogram_equlization/blob/master/pic/Test(3).jpg"  alt="test2" title="GitHub,Social Coding" width="200" height="200" /> 
   
<img src="https://github.com/EsakaKyo/Histogram_equlization/blob/master/pic/NewTest(4).jpg"  align ="left"  alt="test2" title="GitHub,Social Coding" width="200" height="200" />
<img src="https://github.com/EsakaKyo/Histogram_equlization/blob/master/pic/Test(4).jpg"  alt="test2" title="GitHub,Social Coding" width="200" height="200" />
   
<img src="https://github.com/EsakaKyo/Histogram_equlization/blob/master/pic/NewTest(5).jpg"  align ="left"  alt="test2" title="GitHub,Social Coding" width="200" height="200" />
<img src="https://github.com/EsakaKyo/Histogram_equlization/blob/master/pic/Test(5).jpg"  alt="test2" title="GitHub,Social Coding" width="200" height="200" />
