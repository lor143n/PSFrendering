from cmath import exp, pi, sqrt
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


def gaussian_kernel(size, std):
    '''Returns a 2D Gaussian kernel array.'''
    kernel1d = signal.windows.gaussian(size, std=std)
    kernel2d = np.outer(kernel1d, kernel1d)
    return kernel2d / np.sum(kernel2d)


def kernel_db(db_size):
    db = []
    for i in range(db_size):
        if i == 0:
            k = 0.000001
        else:      
            k = i/10
        db.append(gaussian_kernel(7,k))
        
    return db
        
        
            
@njit(parallel = True)
def convolution(rgb, depth, krnls_db, st_pl_depth):
    rgb_new = rgb
    for i in prange(len(rgb)):
        for j in prange(len(rgb[i])):
            
            
            rgb_new[i][j] = (0,0,0) #utilizzo tupla
            krnl_size = 7
            if int(depth[i][j]*10) < len(krnls_db):
                krnl = krnls_db[int(depth[i][j])*10]
            else:
                krnl = krnls_db[len(krnls_db)-1]
                
            if int(depth[i][j]) <= st_pl_depth:
                continue
            

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
                    
    return rgb_new
    
    
def main_convolution():

    (rgb, depth) = load_rgbd('TestImages/Scena Davide/rgbd.exr')

    start_time = time.time()
    krnl_db = kernel_db(1000)
    end_time = time.time()
    print("Convolution time is: ", str(end_time-start_time)+"s")
    start_plane_depth = 2
    
    rgb = convolution(rgb, depth, krnl_db, start_plane_depth)
    end_time = time.time()
    
    save_srgb(rgb, 'ResImages/finalSTD7.png')
    print("Convolution time is: ", str(end_time-start_time)+"s")
    

if __name__=='__main__':
    main_convolution()