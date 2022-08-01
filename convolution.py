import numpy as np
import OpenEXR
import os
import re
import time


from Imath import PixelType
from PIL import Image
from numba import njit, prange
from scipy import signal

    
def load_rgbd(impath):
  E = OpenEXR.InputFile(impath)
  dw = E.header()['dataWindow']
  width = dw.max.x - dw.min.x + 1
  height = dw.max.y - dw.min.y + 1

  rgb = np.zeros((height, width, 3), dtype=np.float32)
  for i, c in enumerate('RGB'):
    buffer = E.channel(c, PixelType(OpenEXR.FLOAT))
    rgb[:, :, i] = np.frombuffer(buffer, dtype=np.float32).reshape(height, width)

  buffer = E.channel('Z', PixelType(OpenEXR.FLOAT))
  depth = np.frombuffer(buffer, dtype=np.float32).reshape(height, width)

  E.close()
  return rgb, depth

def save_srgb(rgb, outpath):
  srgb = np.where(rgb <= 0.0031308,
                  12.92 * rgb,
                  (1 + 0.055) * np.power(rgb, 1 / 2.4) - 0.055)
  srgb = np.clip(srgb * 255, 0, 255).astype(np.uint8)
  Image.fromarray(srgb).save(outpath)
  
  
def gaussian_kernel(size=21, std=3):
    '''Returns a 2D Gaussian kernel array.'''
    kernel1d = signal.windows.gaussian(size, std=std)
    kernel2d = np.outer(kernel1d, kernel1d)
    return kernel2d / np.sum(kernel2d)
            

@njit(parallel=True)
def convolution(rgb, depth, krnl):
    rgb_new = rgb
    for i in prange(len(rgb)):
        for j in prange(len(rgb[i])):
            rgb_new[i][j] = [0,0,0]
            krnl_size = 21; #kernel dispari
            krnl_range = int((krnl_size - 1) / 2)
            
            krnl_i = 0
            for k in prange(i-krnl_range, i+krnl_range+1):
                
                if k < 0 or k >= len(rgb):
                    krnl_i += 1
                    continue
                
                krnl_j = 0
                for h in prange(j-krnl_range, j+krnl_range+1):
                    
                    if h < 0 or h >= len(rgb[i]):
                        krnl_j += 1
                        continue
                    
                    rgb_new[i][j] += rgb[k][h] * krnl[krnl_i][krnl_j]
                    krnl_j += 1
                krnl_i += 1
            #rgb_new[i][j] = (rgb_new/(krnl_size*krnl_size))*krnl_range
                    
    return rgb_new
    
    
def main_convolution():

    (rgb, depth) = load_rgbd('TestImages/Scena Davide/rgbd.exr')
    kernel = gaussian_kernel()

    start_time = time.time()
    rgb = convolution(rgb, depth, kernel)
    end_time = time.time()
    
    save_srgb(rgb, 'ResImages/final.png')
    print("Convolution time is: ", str(end_time-start_time)+"s")
    

if __name__=='__main__':
    main_convolution()