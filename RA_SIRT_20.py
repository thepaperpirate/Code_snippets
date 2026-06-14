import os
import h5py
import numpy as np
import astra
import skimage as ski
from skimage import io
import tqdm 
from tqdm import tqdm
import h5py
import tifffile as tiff
import ants
from scipy.ndimage import shift
from multiprocessing import Pool, TimeoutError
import matplotlib.pyplot as plt
import time
from scipy.ndimage import convolve

class FP_Runner:
    def __init__(self , angles : np.float32 , det_width : int ,  det_spacing : float =1.0):
        self.proj_geom = astra.create_proj_geom('parallel',det_spacing, det_width ,angles)
        self.vol_geom = astra.create_vol_geom(det_width,det_width)
        self.projection_id = astra.data2d.create('-sino', self.proj_geom)
        self.reconstruction_id = astra.data2d.create('-vol',self.vol_geom,np.zeros((det_width,det_width),np.float32))
        cfg = astra.astra_dict('FP_CUDA')
        cfg['ProjectionDataId'] =  self.projection_id
        cfg['VolumeDataId'] = self.reconstruction_id
        self.algorithm_id = astra.algorithm.create(cfg)
    
    def run(self , reconstruction):
        reconstruction32 = np.asarray(reconstruction,np.float32)
        astra.data2d.store(self.reconstruction_id, reconstruction32)
        astra.algorithm.run(self.algorithm_id)
        return(astra.data2d.get(self.projection_id))
    
    def close(self):
        astra.data2d.delete([self.projection_id, self.reconstruction_id])
        astra.algorithm.delete(self.algorithm_id)  

class SIRT_2D_Runner:
    def __init__(self , angles : np.float32, det_width : int ,  det_spacing: float = 1.0):
        
        N_Angles = angles.shape[0]
        
        self.proj_geom = astra.create_proj_geom('parallel', det_spacing, det_width , angles)
        self.vol_geom = astra.create_vol_geom(det_width, det_width)
        self.sino_id = astra.data2d.create('-sino', self.proj_geom , np.zeros((N_Angles , det_width) , np.float32))
        self.recon_id =  astra.data2d.create('-vol', self.vol_geom)
        
        cfg = astra.astra_dict('SIRT_CUDA')
        cfg['ProjectionDataId'] = self.sino_id
        cfg['ReconstructionDataId']= self.recon_id
        self.algorithm_id = astra.algorithm.create(cfg)
    
    def run(self,sino, rec_init, iterations):
        astra.data2d.store(self.recon_id, rec_init)
        sino32 = np.asarray(sino,dtype=np.float32)
        astra.data2d.store(self.sino_id, sino32) 
        astra.algorithm.run(self.algorithm_id, iterations)
        return astra.data2d.get(self.recon_id)
    
        
    def close (self):
        astra.data2d.delete([self.sino_id,  self.recon_id])
        astra.algorithm.delete(self.algorithm_id)

def norm_im(p):
    p= np.asarray(p, dtype = np.float32)
    p -= p.min()
    mx = p.max()
    if mx == 0:
        return np.zeros_like(p, dtype = np.float32)
    else:
        p /= mx
        return p
    
def sharpness(image):
    Kernel=  np.array([
                    [0.0 ,-1.0 ,0.0],
                    [-1.0, 5.0, -1.0],
                    [0.0, -1.0, 0.0]
    ],dtype= np.float32)
    
    return convolve(image.astype(np.float32),Kernel, mode = 'nearest') 

def slice_to_shape(slice_tuple, full_shape):
    shape = []
    for slc, dim in zip(slice_tuple, full_shape):
        if slc.start is None:
            start = 0
        else:
            start = slc.start
        if slc.stop is None:
            stop = dim
        else:
            stop = slc.stop
        shape.append(stop - start)
    return tuple(shape)    

def get_shift_parameters(fixed_image, moving_image):
    pad = 100
    ants_mask = np.zeros_like(fixed_image , dtype = np.float32)
    ants_mask[pad:-pad, pad:pad] = 1
    ants_mask = ants.from_numpy(ants_mask)
    fixed_image_norm = sharpness(norm_im(fixed_image))
    moving_image_norm = sharpness(norm_im(moving_image))
    fixed_image_ants = ants.from_numpy(np.asarray(fixed_image_norm , dtype=np.float32))
    moving_image_ants = ants.from_numpy(np.asarray(moving_image_norm, dtype = np.float32))
    reg = ants.registration(fixed_image_ants , moving_image_ants , type_of_transform='Translation', aff_sampling = 60, aff_random_sampling_rate = 0.6, mask= ants_mask , mask_all_stages=True)
    tx = ants.read_transform(reg["fwdtransforms"][0])
    dy = np.float32(tx.parameters[4])
    dx = np.float32(tx.parameters[5])
    return dy , dx

def get_shifted_image(args):
    projection = args[0]
    dy = args[1]
    dx = args[2]
    shifted = shift(projection,(dy,dx), mode='nearest')
    return shifted.astype(np.float32)

def placeholder(args):
    
    return get_shift_parameters(args[1],args[0])

def pad_projections(arr : np.array , padding : dict):
    if arr.ndim ==2:
        return np.pad(arr, ((padding, padding),(padding , padding)))
    elif arr.ndim == 3:
        return np.pad(arr, ((0,0),(padding,padding),(padding,padding)))



def reprojection_alignment(
read_path : str , 
write_path :str , 
dset_read_data_name : str , 
dset_read_angles_name :  str = "angles", 
downsample : int = 1 , 
verbose : bool = False, 
rec_vol : tuple[slice , slice, slice] = (slice(None),slice(None),slice(None)) , 
max_shift : float = 300.00,
n_iters : int = 6,
use_prior_reco_as_init: bool = True,
):
    
    def main():
        h5_path = write_path

        file = h5py.File(read_path,"r")
        scans = file[dset_read_data_name]
        angles =  np.asarray(file[dset_read_angles_name],dtype=np.float32).reshape(-1)
        angles = np.deg2rad(angles)

        scans = pad_projections(scans, 40)

        reprojections = np.zeros_like(scans, dtype = np.float32)
        projections = np.array(scans , dtype = np.float32)
        
        Runner_Sirt = SIRT_2D_Runner(angles, scans.shape[2])
        Runner_FP = FP_Runner(angles, projections.shape[2])

        ds_arr = np.full((n_iters + 1 , scans.shape[0]), downsample , dtype= np.int32)
        rmse_arr = np.zeros((n_iters, scans.shape[0]),dtype= np.float32)

        M_shifts_x = np.zeros((n_iters, scans.shape[0]), dtype= np.float32)
        delta_M_shifts_x = np.zeros((n_iters, scans.shape[0]), dtype= np.float32) 
        sum_M_shifts_x = np.zeros((n_iters, scans.shape[0]), dtype= np.float32)
        M_shifts_y = np.zeros((n_iters, scans.shape[0]), dtype= np.float32)
        delta_M_shifts_y = np.zeros((n_iters, scans.shape[0]), dtype= np.float32)
        sum_M_shifts_y = np.zeros((n_iters , scans.shape[0]), dtype= np.float32)

        recons = np.zeros((projections.shape[1],projections.shape[2],projections.shape[2]),dtype= np.float32)
        for iteration in tqdm(range(n_iters)):
            
            with h5py.File(h5_path, "a") as h5:
                dset_rec_name = "reconstruction_iter_" + str(0) + str(iteration)
                if dset_rec_name in h5:
                    del h5[dset_rec_name]
                #recon_shape = (projections.shape[1], *slice_to_shape(rec_vol, (projections.shape[2],projections.shape[2])))
                #recons = np.zeros(recon_shape,dtype = np.float32)


                for i in tqdm(range(projections.shape[1]), desc= "SIRT + FP" , disable = not verbose):
                    if use_prior_reco_as_init == True:
                        if iteration == 0:
                            SIRT_iter = 2
                            rec_init = 0
                        elif iteration > 0 :
                            SIRT_iter = 1
                            rec_init = recons[i]

                    sino = projections[: , i , :]
                    rec = Runner_Sirt.run(sino, rec_init, SIRT_iter)
                    
                    recons[i,:,:] = rec
                    
                    reprojection = Runner_FP.run(rec)
                    reprojections[:,i,:] = reprojection
                #recons_transposed = np.transpose(recons[rec_vol],(1,0,2))
                recons_transposed = recons
                print("reconstruction + reprojection completed")
              
                if iteration == 5 or iteration == 10 or iteration == 15 or iteration == 20 or iteration == 25 or iteration == 30 or iteration == 35:
                    h5.create_dataset(
                        name = dset_rec_name,
                        data = recons_transposed,
                        dtype = np.float32,
                        chunks = (1, recons_transposed.shape[1], recons_transposed.shape[2])
                    )
                projections_save = h5.require_group("projections_save")
                projections_save.create_dataset(
                    name = f"projections_iteration_0{iteration}",
                    data = projections,
                    dtype = np.float32
                )
                projections_save.create_dataset(
                    name = f"reprojections_interation_0{iteration}",
                    data = reprojections,
                    dtype = np.float32
                )
                
            projections_ds = []
            reprojections_ds = []

            #for i in range(projections.shape[0]): 
            #    if iteration <= 5:                        #!!! short term sanity change !!!
            #        ds = ds_arr[iteration, i]
            #    elif iteration <= 10:
            #        ds = 4
            #        ds_arr[iteration,:] = ds
            #    else:
            #        ds = 1
            #        ds_arr[iteration,:] = ds
            #
            ds = 1
            for i in range(projections.shape[0]):
                projections_ds.append(projections[i, ::ds, ::ds])
                reprojections_ds.append(reprojections[i, ::ds, ::ds])
            

            
            with Pool(processes= 2) as pool:
                args = [(reprojections_ds[i] , projections_ds[i] , ds_arr[iteration , i]) for i in range(projections.shape[0])]
                shifts_ds = np.array(pool.map(placeholder,args))  
            
            shifts = shifts_ds * downsample

            print("getting shifts completed")
            
            M_shifts_x[iteration , : ] = shifts[:, 1].reshape(1, -1)
            M_shifts_y[iteration , :] = shifts[:,0].reshape(1, -1)

            if iteration >=1 :
                delta_M_shifts_x[iteration , :] = (M_shifts_x[iteration , :]) - (M_shifts_x[iteration - 1, :])
                delta_M_shifts_y[iteration , :] = (M_shifts_y[iteration , :]) - (M_shifts_y[iteration - 1, :])
                
                sum_M_shifts_x[iteration , :] = sum_M_shifts_x[iteration -1 , :] + np.clip(M_shifts_x[iteration , :], -max_shift , max_shift)
                sum_M_shifts_y[iteration , :] = sum_M_shifts_y[iteration -1 , :] + np.clip(M_shifts_y[iteration , :], -max_shift , max_shift)  
                
            else: 
                delta_M_shifts_x[iteration , :] = 0
                delta_M_shifts_y[iteration , :] = 0

                sum_M_shifts_x[iteration , :] = np.clip(M_shifts_x[iteration , :], -max_shift , max_shift)
                sum_M_shifts_y[iteration , :] = np.clip(M_shifts_y[iteration , :], -max_shift , max_shift)

            #for i in range(scans.shape[0]):
            #    if np.abs(delta_M_shifts_x[iteration, i]) < threshold_1 and np.abs(delta_M_shifts_y[iteration , i]) < threshold_1 : 
            #        ds_arr[iteration + 1 , i] = np.round(downsample / 2)
            #        print("ds_1_done ", iteration, " " , i )
            #    if np.abs(delta_M_shifts_x[iteration, i]) < threshold_2 and np.abs(delta_M_shifts_y[iteration, i]) < threshold_2:
            #        ds_arr[iteration + 1 , i] = 1
            #        print("ds_removed ",iteration, " ", i)
            
            if verbose == True: 
                print("shift mean abs:", np.mean(np.abs(shifts), axis=0), " shift max abs:", np.max(np.abs(shifts), axis=0), "allowed max shift:", max_shift)
            
            
            
            with Pool(processes = 4) as pool:   
                args = [(scans[i], sum_M_shifts_y[iteration , i], sum_M_shifts_x[iteration , i]) for i in range(scans.shape[0])] 
                for i, img in enumerate(
                    
                    pool.map(get_shifted_image, args)
                    
                ):
                    projections[i] = img
            
            for i in range(projections.shape[0]):
                rmse= np.sqrt(np.mean((projections[i] - reprojections[i])**2))
                rmse_arr[iteration, i] = rmse
        
        with h5py.File(h5_path , "a") as h5:
            dset_name = "RMSE_iter_"+ str(n_iters)
            h5.create_dataset(
                name= dset_name,
                data= rmse_arr
            )
            shifts = h5.require_group("shifts")

            M_x_name = "shift_array_x" + str(n_iters)
            if M_x_name in shifts:
                del shifts[M_x_name]
            shifts.create_dataset(
                name = M_x_name,
                data = M_shifts_x,
                dtype = np.float32
            )
            M_y_name = "shift_array_y" + str(n_iters)
            if M_y_name in shifts:
                del shifts[M_y_name]
            shifts.create_dataset(
                name = M_y_name,
                data = M_shifts_y,
                dtype= np.float32
            )
            sum_y_name = "sum_shift_array_y" + str(n_iters)
            if sum_y_name in shifts:
                    del shifts[sum_y_name]
            shifts.create_dataset(
                name = sum_y_name,
                data = sum_M_shifts_y,
                dtype = np.float32
            )
            sum_x_name = "sum_shift_array_x" + str(n_iters)
            if sum_x_name in shifts:
                del shifts[sum_x_name]
            shifts.create_dataset(
            name = sum_x_name,
            data = sum_M_shifts_x,
            dtype = np.float32
            )
            
        
        if verbose == True:
            print(rmse_arr)
            #shifts_x = shifts[:,1]
            #shifts_y = shifts[:,0]
            #x = np.arange(shifts.shape[0])
            #plt.plot(x, shifts_x, label = "X_shifts")
            #plt.plot(x, shifts_y, label = "Y_shifts")
            #plt.xlabel("projektionen")
            #plt.ylabel("shifts")
            #plt.legend()
            #plt.show()

    if __name__ == '__main__':

        main()
    return "You can find your reconstructions at: ",  write_path

print(reprojection_alignment(
    read_path = "/home/lewitsch/lewits_dust/tomography/chip_alignment/projection_stacks/CoR_corr_phase_refracted.h5", 
    write_path= "/asap3/petra3/gpfs/p06/2026/data/11022533/scratch_cc/analysis/Reprojection_Alignment/Results/RA_microchip_20.h5", 
    dset_read_data_name="scans_COR_corr" , 
    dset_read_angles_name="angles", 
    verbose = False, 
    rec_vol = (slice(None), slice(1500,2800),slice(None)), 
    n_iters = 40,
    use_prior_reco_as_init = True
    ))