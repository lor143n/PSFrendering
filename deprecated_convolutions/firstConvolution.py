from numba import njit
import imagesManager as imaMan
import math
import time


def kernel_db_std(db_size, k_size):
    db = []
    for i in range(db_size):
        db.append(imaMan.gaussian_kernel(k_size,i/10 + 0.1))
    return db

def kernel_db_size(db_size):
    db = []
    disp_i = 1
    for i in range(db_size):
        db.append(imaMan.gaussian_kernel(disp_i,1))
        disp_i = disp_i + 2
        
    return db

def kernel_db_size_std(db_size):
    db = []
    disp_i = 1
    for i in range(db_size):
        # [0.0,99.9] db.append(gaussian_kernel(disp_i,i/10))
        # [0,999] db.append(gaussian_kernel(disp_i,i))
        db.append(imaMan.gaussian_kernel(disp_i, math.e**(i/5)))
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
            
            
            #Calcolo del circle of confusion
            #85mm to m  
            focal_length = 50.0 / 1000
            
            f = 1/focal_length + 1/focus
            magnification = f / (focus - f)
            Aperture = 1.8
            
            CoC = round(Aperture * magnification * (abs(depth[i][j] - focus) / depth[i][j]))
            
            
            if CoC < len(krnls_db):
                krnl = krnls_db[CoC]
            else:
                krnl = krnls_db[len(krnls_db)-1]
        
            
            for x in range(krnl_size):
                for y in range(krnl_size):
                    rgb_new[i][j] += krnl[x][y] * rgb[i-krnl_size+x][j-krnl_size+y]
            
                    
    return rgb_new


def main():

    
    #Caricamento dell'immagine exr
    #(rgb, depth) = load_rgbd('TestImages/Scena Davide/rgbd.exr')
    (rgb, depth) = imaMan.load_rgbd('TestImages/blackfield1024_100.exr')
    
    start_time = time.time()
    
    #Creazione del database di kernel
    krnl_db = kernel_db_std(10000, 7)
    #krnl_db = kernel_db_size(20)
    #krnl_db = kernel_db_size_std(10)
    
    end_time = time.time()
    print("Database construction time is: ", str(end_time-start_time)+"s")
    
    #Convoluzione
    rgb = convolution(rgb, depth, krnl_db, len(krnl_db[0]), 1)
    
    end_time = time.time()
    print("Convolution time is: ", str(end_time-start_time)+"s")
    
    #Salvataggio dell'immagine
    imaMan.save_srgb(rgb, 'ResImages/blackfield2.png')
    
        

if __name__=='__main__':
    main()