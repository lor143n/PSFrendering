import numpy as np
import OpenEXR
import os
import re

from functools import partial
from Imath import PixelType
from PIL import Image
from multiprocessing import cpu_count, Pool
from multiprocessing.pool import ThreadPool
from numba import njit, prange
from numpy.lib.stride_tricks import sliding_window_view
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from scipy.ndimage import center_of_mass, shift, zoom
from tqdm import tqdm

#aprire file exr in python
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

#salvare matrice rgb manipolata in png
def save_srgb(rgb, outpath):
  srgb = np.where(rgb <= 0.0031308,
                  12.92 * rgb,
                  (1 + 0.055) * np.power(rgb, 1 / 2.4) - 0.055)
  srgb = np.clip(srgb * 255, 0, 255).astype(np.uint8)
  Image.fromarray(srgb).save(outpath)


#salvare matrice rgb manipolata in exr (bianco e nero)
def save_exr(img, outpath):
  height, width = img.shape
  header = OpenEXR.Header(width, height)
  header['channels'] = {'Y': Channel(PixelType(OpenEXR.FLOAT))}

  exr = OpenEXR.OutputFile(outpath, header)
  exr.writePixels({'Y': img.tobytes()})
  exr.close()

#creazione della matrice gaussiana
https://stackoverflow.com/questions/29731726/how-to-calculate-a-gaussian-kernel-matrix-efficiently-in-numpy
def gaussian_kernel(size=21, std=3):
    '''Returns a 2D Gaussian kernel array.'''
    kernel1d = signal.windows.gaussian(size, std=std)
    kernel2d = np.outer(kernel1d, kernel1d)
    return kernel2d / np.sum(kernel2d)
