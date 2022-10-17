import numpy as np
import OpenEXR
from Imath import PixelType
from PIL import Image
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

def load_psf(impath):

    E = OpenEXR.InputFile(impath)
    dw = E.header()['dataWindow']
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1

    buffer = E.channel('Y', PixelType(OpenEXR.FLOAT))
    depth = np.frombuffer(buffer, dtype=np.float32).reshape(height, width)

    E.close()
    return depth

def save_srgb(rgb, outpath):
    srgb = np.where(rgb <= 0.0031308,
                  12.92 * rgb,
                  (1 + 0.055) * np.power(rgb, 1 / 2.4) - 0.055)
    srgb = np.clip(srgb * 255, 0, 255).astype(np.uint8)
    Image.fromarray(srgb).save(outpath)
  
  
def save_exr(rgb, outpath):
    r, g, b = np.split(rgb, 3, axis=-1)
    height, width = rgb.shape[:-1]
    header = OpenEXR.Header(width, height)
	
    exr = OpenEXR.OutputFile(outpath, header)
    exr.writePixels({'R': r.tobytes(),
	                 'G': g.tobytes(),
	                 'B': b.tobytes()})
    exr.close()
    
def save_exr_psf(rgb, outpath):
    r = np.split(rgb, 1, axis=-1)
    height, width = rgb.shape[:-1]
    header = OpenEXR.Header(width, height)
	
    exr = OpenEXR.OutputFile(outpath, header)
    exr.writePixels({'Y': r.tobytes()})
    exr.close()

  
def gaussian_kernel(size, std):
    '''Returns a 2D Gaussian kernel array.'''
    kernel1d = signal.windows.gaussian(size, std=std)
    kernel2d = np.outer(kernel1d, kernel1d)
    return kernel2d / np.sum(kernel2d)
