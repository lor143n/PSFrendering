from array import array
from curses.panel import top_panel
from importlib.resources import path
import math
from numba import njit, prange, float64
from numpy import float64, ix_, pad
from scipy import ndimage as nim
from PIL import Image
import imagesManager as imaMan
import time
import os



def upload_PSF(depth, xy, db, focus):
    
    # Caricamento dell PSF
    
    return 


def kernel_db_std(depths_count, focus, k_size, aperture, focal_length):
    
    db = []
    N = (focal_length)/aperture 
    
    for i in range(1,depths_count):
        CoC = (abs((focus-i) / i)) * ((focal_length**2)/N * abs(focus-focal_length)) * 1000 #to mm
        if i == focus:
            CoC += 0.01
        db.append(imaMan.gaussian_kernel(k_size, CoC))
    
    return db

def psf_db(krnl_size):
    
    camera_path = '/home/lor3n/Documents/GitHub/PFSrendering/psf/petzval/focus-5.00m/aperture-f1'
    krnl_range = int((krnl_size - 1) / 2)
    
    for directory in os.listdir(camera_path):
        
        dir_path = os.path.join(camera_path, directory)
        for file in os.listdir(dir_path):
            
            file_path = os.path.join(dir_path, file)
            big_psf = imaMan.load_psf(file_path)
            ascom = nim.center_of_mass(big_psf)
            com = (round(ascom[0]), round(ascom[1]))
            
            
            big_psf_pad = pad(big_psf, (krnl_range,))
            
            print(str(com[0])+' : '+str(com[0] + krnl_range)+' , '+str(com[1]-krnl_range)+' : '+str(com[1]+krnl_range))
            ker_psf = big_psf_pad[com[0] : com[0] + (2*krnl_range)+1 , com[1] : com[1]+ (2*krnl_range)+1]
            imaMan.save_srgb(ker_psf, "testPSf/"+directory+file+" : "+str(com[0])+"-"+str(com[1])+".png")
            
            
    return

   

@njit(parallel = True)
def psf_convolution(rgb, depth, krnls_db, focus):
    
    rgb_new = rgb*0
    krnl_size = len(krnls_db[0])
    krnl_range = int((krnl_size - 1) / 2)
    
    #aggiungere padding
    
    image_width = len(rgb)
    image_height = len(rgb[0])
    
    for i in prange(krnl_range, image_width - krnl_range):
        for j in range(krnl_range, image_height - krnl_range):
            
            krnl = [0.0]*(krnl_size**2)
            kernelSum = 0
    
            for h in range(krnl_size):
                for k in range(krnl_size):
                       
                       
                    dep = round(depth[i][j])
                    pos = 0
                                 
                    if dep < len(krnls_db):
                        psfij = krnls_db[dep]
                    else:
                        psfij = krnls_db[len(krnls_db)-1]
                    
                    ijvalue = psfij[h][k]
                    krnl[h*krnl_size + k] = ijvalue
                    kernelSum += ijvalue
            
            for x in range(krnl_size):
                for y in range(krnl_size):
                    rgb_new[i][j] += krnl[x*krnl_size+y] * rgb[i-krnl_size+x][j-krnl_size+y]
                      
    return rgb_new
       

    

def main():
    
    '''
    ker_size = 17
    focus = 5
    aperture = 50 #mm
    focal_length = 50 #mm 
    
    #PETZVAL FOCUS-5.0m aperture f/1
    
    export = input('Export type: (1) exr (2) png\n')
    while export != '1' and export != '2' :
        print("Type 1 or 2")
        export = input('Export type: (1) exr (2) png\n')
    export = int(export)
    print('Algorithm start')
    
    (rgb, depth) = imaMan.load_rgbd('TestImages/water2.exr')
    
    
    start_time = time.time()
    
    #Gaussian kernels database with std from 0.1 to 10.00
    #Gaussian kernels with size between 0 and 15 has no changes with std grater than 10.00
    #depths_count must be greater than focus level
    krnl_db = kernel_db_std(100, focus, ker_size, aperture / 1000, focal_length / 1000)

    db_end_time = time.time()
    
    rgb = psf_convolution(rgb, depth, krnl_db, focus)
    
    conv_end_time = time.time()
    
    print("Database construction time is: ", str(db_end_time-start_time)+"s")
    print("Convolution time is: ", str(conv_end_time-start_time)+"s")
    
    if export == 1:
        imaMan.save_exr(rgb, 'ResImages/water_size['+str(ker_size)+']foc['+str(focus)+']foc_length['+str(focal_length)+']f-stop['+str(focal_length/aperture)+'].exr')
    else:
        imaMan.save_srgb(rgb, 'ResImages/water_size['+str(ker_size)+']foc['+str(focus)+']foc_length['+str(focal_length)+']f-stop['+str(focal_length/aperture)+'].png')
    '''
    
    
    psf_db(13)
    
if __name__=='__main__':
    main()