import numpy as np
import OpenEXR
import os
from Imath import PixelType, Channel
from scipy import ndimage as nim
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
    
def save_exr_psf(depth, outpath):

    height, width = depth.shape

    header = OpenEXR.Header(width, height)
    header['channels'] = {'Y': Channel(PixelType(OpenEXR.FLOAT))}
    exr = OpenEXR.OutputFile(outpath, header)
    exr.writePixels({'Y': depth.tobytes()})

    exr.close()

  
def gaussian_kernel(size, std):
    '''Returns a 2D Gaussian kernel array.'''
    kernel1d = signal.windows.gaussian(size, std=std)
    kernel2d = np.outer(kernel1d, kernel1d)
    return kernel2d / np.sum(kernel2d)



#DEPRECATED CODE


def psf_db(krnl_size, psf_dim, img_width, camera_path):
    
    krnl_range = int((krnl_size - 1) / 2)
    
    psf_dict = []
    dict_index = 0
    
    for directory in os.listdir(camera_path):
        dir_clean = directory.split('-')
        dir_depth = float(dir_clean[1].split('m')[0])        
        print(dir_depth)
        psf_dict.append( (dir_depth, []) )
        
        dir_path = os.path.join(camera_path, directory)
        for file in os.listdir(dir_path):
            
            file_path = os.path.join(dir_path, file)
            big_psf = load_psf(file_path)
            
            
            com = nim.center_of_mass(big_psf)
            
            shift_x, center_x = np.modf(com[0])
            shift_y, center_y = np.modf(com[1])
            
            center_x = int(center_x)
            center_y = int(center_y)
            
            big_psf_pad = np.pad(big_psf, ((krnl_range, krnl_range), (krnl_range, krnl_range)))
            
            ker_psf = big_psf_pad[center_x : center_x + (2*krnl_range)+1 , center_y : center_y + (2*krnl_range)+1]
            
            ker_psf = nim.shift(ker_psf, (shift_x, shift_y))
            
            psf_dict[dict_index][1].append( (com[0],com[1],ker_psf) )
            
        dict_index += 1
            
    psf_dict.sort()
    return psf_dict
