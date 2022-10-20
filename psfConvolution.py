from array import array
from curses.panel import top_panel
from importlib.resources import path
import math
from select import select
from numba import njit, prange, float64
import numpy as np
from scipy import ndimage as nim
from PIL import Image
import imagesManager as imaMan
import time
import os
import re



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
            big_psf = imaMan.load_psf(file_path)
            
            
            com = nim.center_of_mass(big_psf)
            zoom_adj = psf_dim / img_width
            
            shift_x, center_x = np.modf(com[0])
            shift_y, center_y = np.modf(com[1])
            
            center_x = int(center_x)
            center_y = int(center_y)
            
            big_psf_pad = np.pad(big_psf, ((krnl_range, krnl_range), (krnl_range, krnl_range)))
            
            ker_psf = big_psf_pad[center_x : center_x + (2*krnl_range)+1 , center_y : center_y + (2*krnl_range)+1]
            
            ker_psf = nim.shift(ker_psf, (shift_x, shift_y))
            
            psf_dict[dict_index][1].append( (com[0],com[1],ker_psf) )
            
            #imaMan.save_srgb(ker_psf, "testPSf/"+directory+file+" : "+str(com[0])+"-"+str(com[1])+".png")
        dict_index += 1
            
    psf_dict.sort()
    return psf_dict
  

@njit()
def psf_convolution(rgb, res, depth, krnls_db, focus, camera_depths):
    
    rgb_new = res
    krnl_size = 13
    krnl_range = int((krnl_size - 1) / 2)
    
    #aggiungere padding
    
    image_width = len(rgb)
    image_height = len(rgb[0])
    
    for i in range(krnl_range, image_width - krnl_range): #p
        print(i)
        for j in range(krnl_range, image_height - krnl_range):
            
            krnl = [0.0]*(krnl_size**2)
            kernelSum = 0
    
            for h in range(krnl_size):
                for k in range(krnl_size):
                       
                       
                    #SELECTING DEPTH   
                    dep = float(round(depth[i][j]*10)/10)
                    chosen_dep_index = 0
                    
                    for dp_index in range(len(krnls_db)):
                        
                        if dp_index == len(krnls_db)-1:
                            chosen_dep_index = len(krnls_db)-1
                            break
                        
                        if dep >= krnls_db[dp_index][0] and dep <= krnls_db[dp_index+1][0]:
                             
                            diff1 = dep - krnls_db[dp_index][0]
                            diff2 = krnls_db[dp_index+1][0] - dep
                             
                            if diff1 <= diff2:
                                chosen_dep_index = dp_index
                            else:
                                chosen_dep_index = dp_index+1
                            
                            break
                    
                    pos_db = krnls_db[chosen_dep_index][1]
                    
                    #SELECTED DEPTH
                    #SELECTING POSITION
                    
                    min_distance = 10000
                    selected_krnl = pos_db[0][2]
                    
                    for psf in pos_db:
                        dist = math.sqrt((i-psf[0])**2 + (j-psf[1])**2)
                        if dist < min_distance:
                            min_distance = dist
                            selected_krnl = psf[2]
                    
                    
                    #POSITION SELECTED
                    
                    ijvalue = selected_krnl[h][k]
                    #ijvalue = psfij[h][k]
                    krnl[h*krnl_size + k] = ijvalue
                    kernelSum += ijvalue
            
            
            
            #NORMALIZATION
            for elem in range(len(krnl)):
                krnl[elem] /= kernelSum
            
            
            for x in range(krnl_size):
                for y in range(krnl_size):
                    rgb_new[i-krnl_range][j-krnl_range] += krnl[x*krnl_size+y] * rgb[i-krnl_size+x][j-krnl_size+y]
            
                      
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
    
    rgb = psf_convolution(rgb, depth, krnl_db, focus, camera_depths)
    
    conv_end_time = time.time()
    
    print("Database construction time is: ", str(db_end_time-start_time)+"s")
    print("Convolution time is: ", str(conv_end_time-start_time)+"s")
    
    if export == 1:
        imaMan.save_exr(rgb, 'ResImages/water_size['+str(ker_size)+']foc['+str(focus)+']foc_length['+str(focal_length)+']f-stop['+str(focal_length/aperture)+'].exr')
    else:
        imaMan.save_srgb(rgb, 'ResImages/water_size['+str(ker_size)+']foc['+str(focus)+']foc_length['+str(focal_length)+']f-stop['+str(focal_length/aperture)+'].png')
    '''
    
    ker_size = 13
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
    
    camera_path = '/home/lor3n/Documents/GitHub/PFSrendering/psf/petzval/focus-5.00m/aperture-f1'
    
    camera_depths = []
    
    for directory in os.listdir(camera_path):
        
        dir_clean = directory.split('-')
        dir_depth = float(dir_clean[1].split('m')[0])        
        camera_depths.append(dir_depth)
        
    camera_depths.sort()
    
    (rgb, depth) = imaMan.load_rgbd('TestImages/1024tree.exr')
    
    start_time = time.time()
    
    krnl_db = psf_db(ker_size, 1024, len(rgb[0]), camera_path)
    
    db_end_time = time.time()
    print("Database construction time is: ", str(db_end_time-start_time)+"s")
    
    start_time = time.time()
    
    krnl_range = int((ker_size - 1) / 2)
    rgb_res = rgb*0
    
    rgb = np.pad(rgb, ((krnl_range, krnl_range), (krnl_range, krnl_range), (0, 0)))
    depth = np.pad(depth, ((krnl_range, krnl_range), (krnl_range, krnl_range)))
    
    print(str(len(rgb))+" - "+str(len(rgb[0])))
    print(str(len(rgb_res))+" - "+str(len(rgb_res[0])))
    
    rgb_res = psf_convolution(rgb, rgb_res, depth, krnl_db, focus, camera_depths)
    
    conv_end_time = time.time()
    
    print("Convolution time is: ", str(conv_end_time-start_time)+"s")
    
    if export == 1:
        imaMan.save_exr(rgb_res, 'ResImages/treePADzero['+str(ker_size)+']foc['+str(focus)+']foc_length['+str(focal_length)+']f-stop['+str(focal_length/aperture)+'].exr')
    else:
        imaMan.save_srgb(rgb_res , 'ResImages/treePADzero['+str(ker_size)+']foc['+str(focus)+']foc_length['+str(focal_length)+']f-stop['+str(focal_length/aperture)+'].png')
        

    
if __name__=='__main__':
    main()