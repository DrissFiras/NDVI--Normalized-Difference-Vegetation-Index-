import cv2
import numpy as np
import matplotlib.pyplot as plt

color_image=cv2.imread('ndvi_n_r.jpg')
noir_image=cv2.imread('ndvi_i_r.jpg')

red_channel = color_image[:,:,0]  
nir_channel = noir_image[:,:,0]

warp_mode = cv2.MOTION_TRANSLATION

if warp_mode == cv2.MOTION_HOMOGRAPHY :   
     warp_matrix = np.eye(3, 3, dtype=np.float32)  
else :  
     warp_matrix = np.eye(2, 3, dtype=np.float32)

# Specify the number of iterations.  
number_of_iterations = 5000;  
  
# Specify the threshold of the increment  
# in the correlation coefficient between two iterations   
termination_eps = 1e-10;  
  
# Define termination criteria  
criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)

# Run the ECC algorithm. The results are stored in warp_matrix.  

(cc, warp_matrix) = cv2.findTransformECC (red_channel,nir_channel,warp_matrix, warp_mode, criteria)

if warp_mode == cv2.MOTION_HOMOGRAPHY :  

    # Use warpPerspective for Homography   
    nir_aligned = cv2.warpPerspective (nir_channel, warp_matrix, (640,480), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    red_aligned=cv2.warpPerspective (red_channel, warp_matrix, (640,480), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
else :  

    # Use warpAffine for nit_channel, Euclidean and Affine  
    nir_aligned = cv2.warpAffine(nir_channel, warp_matrix, (640,480), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP);  
    red_aligned = cv2.warpAffine(red_channel, warp_matrix, (640,480), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP);  


cv2.imwrite('NIRALIGNED.png',nir_aligned)
cv2.imwrite('redaligned.png',red_aligned)


ligne,colonne=nir_aligned.shape
ndvi=np.empty((ligne,colonne), dtype=np.float32)
for i in range(ligne):
    for j in range(colonne):
         if (float(nir_aligned[i,j]) + float(red_channel[i,j]) == 0):
             ndvi[i,j]=0
         else : ndvi[i,j]=float(float(nir_aligned[i,j])-float(red_channel[i,j]))/float(float(nir_aligned[i,j])+ float(red_channel[i,j]))
print ndvi




f=plt.figure()

plt.imshow(ndvi, cmap='nipy_spectral')
cbar = plt.colorbar(orientation='vertical')
cbar.set_label('NDVI : Indice de Vegetation')
f.savefig('NDVI_3.png')
plt.show()


