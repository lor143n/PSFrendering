from cmath import exp, pi, sqrt
from ctypes import sizeof
import math
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
  
  
def save_exr(img, outpath):
  r, g, b = img
  height, width = img.shape
  header = OpenEXR.Header(width, height)

  exr = OpenEXR.OutputFile(outpath, header)
  exr.writePixels({'R': r.tobytes(),
                   'G': g.tobytes(),
                   'B': b.tobytes()})
  exr.close()


def gaussian_kernel(size, std):
    
    '''Returns a 2D Gaussian kernel array.'''
    kernel1d = signal.windows.gaussian(size, std=std)
    kernel2d = np.outer(kernel1d, kernel1d)
    return kernel2d / np.sum(kernel2d)



def kernel_db_std(db_size, k_size):
    db = []
    for i in range(db_size):
        # [0,999] db.append(gaussian_kernel(k_size,i))
        # [0.0,99.9] db.append(gaussian_kernel(k_size,i/10))
        # [0.00,9.99] db.append(gaussian_kernel(k_size,i/100))
        # [1.00, 10.99] db.append(gaussian_kernel(k_size,i/100 + 1))
        # [0.1, 100.0] db.append(gaussian_kernel(k_size,i/10 + 0.1))
        db.append(gaussian_kernel(k_size,i/10 + 0.1))
        #db.append(gaussian_kernel(k_size,i/10 + 1))
    tuple(map(tuple, db))
    return db

def kernel_db_size(db_size):
    db = []
    disp_i = 1
    for i in range(db_size):
        db.append(gaussian_kernel(disp_i,1))
        disp_i = disp_i + 2
        
    return db

def kernel_db_size_std(db_size):
    db = []
    disp_i = 1
    for i in range(db_size):
        # [0.0,99.9] db.append(gaussian_kernel(disp_i,i/10))
        # [0,999] db.append(gaussian_kernel(disp_i,i))
        db.append(gaussian_kernel(disp_i, math.e**(i/5)))
        disp_i = disp_i + 2
        
    return db
        
        
            
@njit()
def convolution(rgb, depth, krnls_db, krnl_size, focus):
    rgb_new = rgb*0
    
    krnl_range = int((krnl_size - 1) / 2)
    image_width = len(rgb)
    image_height = len(rgb[0])
    
    for i in range(krnl_range, image_width - krnl_range):
        for j in range(krnl_range, image_height - krnl_range):
            
            krnl_depth = round(abs(depth[i][j] - focus))
            
            if krnl_depth**2 < len(krnls_db):
                krnl = krnls_db[krnl_depth**2]
            else:
                krnl = krnls_db[len(krnls_db)-1]
        
            
            for x in range(krnl_size):
                for y in range(krnl_size):
                    rgb_new[i][j] += krnl[x][y] * rgb[i-krnl_size+x][j-krnl_size+y]
            
                    
    return rgb_new
    
    
def main_convolution():


    #Caricamento dell'immagine exr
    (rgb, depth) = load_rgbd('TestImages/Scena Davide/rgbd.exr')
    
    start_time = time.time()
    
    #Creazione del database di kernel
    krnl_db = kernel_db_std(1000, 7)
    #krnl_db = kernel_db_size(20)
    #krnl_db = kernel_db_size_std(10)
    
    end_time = time.time()
    print("Database construction time is: ", str(end_time-start_time)+"s")
    
    #Convoluzione
    print(krnl_db[999])
    rgb = convolution(rgb, depth, krnl_db, len(krnl_db[0]), 11)
    
    end_time = time.time()
    print("Convolution time is: ", str(end_time-start_time)+"s")
    
    #Salvataggio dell'immagine
    save_srgb(rgb, 'ResImages/resGAUSS(17 - 0.1_100)[88]norm.png')
    

if __name__=='__main__':
    main_convolution()