from Histogram_equalization import HistogramEqualization

if __name__ == '__main__':
    HE = HistogramEqualization('./pic/Test.jpg')
    HE.draw_histogram()