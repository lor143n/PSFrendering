from scipy import ndimage as nim
import imagesManager as imaMan
import numpy as np
import decimal
import click
import os

@click.command()
@click.argument('camera_name', default='canon-zoom')
@click.argument('krnl_size', default=13)
@click.argument('focus', default=5.0)
@click.argument('aperture', default=1.4)
def kernelFilesGenerator(camera_name, krnl_size, focus, aperture):
    
    camera_path = f'/home/lor3n/Documents/psf_gen/out/{camera_name}/focus-{focus}0m/aperture-f{aperture}'
    
    camera_new_path = f'/home/lor3n/Documents/GitHub/PSFrendering/PSF_kernels/{camera_name}_{krnl_size}_{focus}_{aperture}'
    os.mkdir(camera_new_path, mode=0o777)
    
    krnl_range = int((krnl_size - 1) / 2)
    
    for directory in os.listdir(camera_path):
        dir_clean = directory.split('-')
        dir_depth = float(dir_clean[1].split('m')[0])    

        directory_new_path = os.path.join(camera_new_path, directory)
        os.mkdir(directory_new_path, mode=0o777)
        dir_path = os.path.join(camera_path, directory)
        
        print(dir_depth)
        for file in os.listdir(dir_path):
            
            file_path = os.path.join(dir_path, file)

            big_psf = imaMan.load_psf(file_path)
            com = nim.center_of_mass(big_psf)
            
            big_psf_pad = np.pad(big_psf, ((krnl_range+1, krnl_range+1), (krnl_range+1, krnl_range+1)))
            
            shift_x, center_x = np.modf(com[0])
            shift_y, center_y = np.modf(com[1])
            center_x = int(center_x) + krnl_range + 1
            center_y = int(center_y) + krnl_range + 1
            
            ker_psf = big_psf_pad[center_x-krnl_range-1 : center_x+krnl_range+2 , center_y-krnl_range-1 : center_y+krnl_range+2]
            
            ker_psf = nim.shift(ker_psf, (-shift_x, -shift_y))
            
            ker_psf = ker_psf[1 : len(ker_psf)-1, 1 : len(ker_psf)-1]
    
            for i in range(len(ker_psf)):
                for j in range(len(ker_psf[i])):
                    ker_psf[i][j] = abs(ker_psf[i][j])
        
            ctx = decimal.Context()
            ctx.prec = 20
            c0 = ctx.create_decimal(repr(com[0]))
            c1 = ctx.create_decimal(repr(com[1]))
        
            file_name = f"{c0}-{c1}"
            file_new_path = os.path.join(directory_new_path, file_name)
            
            
            imaMan.save_exr_psf(ker_psf, file_new_path)


if __name__=='__main__':
    kernelFilesGenerator()
