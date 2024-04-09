import cv2
import numpy as np

class GaborFliter:
    def __init__(self):
        self.filters = self.build_filters()

    def build_filters(self):
        #filters = []
        ksize = 3
        lamda = np.pi / 1.6
        theta = np.pi / 2

        kern = cv2.getGaborKernel((ksize, ksize), 1, theta, lamda, 0.5, 0, ktype=cv2.CV_32F)
        kern /= 1.5 * kern.sum()
        #filters.append(kern)

        return kern#filters

    def getGabor(self, img):
        #res = []
        accum = np.zeros_like(img)
        kern = self.filters[0]
        fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
        heat_img = cv2.applyColorMap(fimg, cv2.COLORMAP_JET)
        accum = np.maximum(accum, heat_img, accum)
        #res.append(np.asarray(accum))
        res=np.asarray(accum)
        return res
