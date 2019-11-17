import cv2
import numpy as np
from skimage import transform as trans

src1 = np.array([
     [51.642,50.115],
     [57.617,49.990],
     [35.740,69.007],
     [51.157,89.050],
     [57.025,89.702]], dtype=np.float32)
#<--left
src2 = np.array([
    [45.031,50.118],
    [65.568,50.872],
    [39.677,68.111],
    [45.177,86.190],
    [64.246,86.758]], dtype=np.float32)

#---frontal
src3 = np.array([
    [39.730,51.138],
    [72.270,51.138],
    [56.000,68.493],
    [42.463,87.010],
    [69.537,87.010]], dtype=np.float32)

#-->right
src4 = np.array([
    [46.845,50.872],
    [67.382,50.118],
    [72.737,68.111],
    [48.167,86.758],
    [67.236,86.190]], dtype=np.float32)

#-->right profile
src5 = np.array([
    [54.796,49.990],
    [60.771,50.115],
    [76.673,69.007],
    [55.388,89.702],
    [61.257,89.050]], dtype=np.float32)

src = np.array([src1,src2,src3,src4,src5])
src_map = {112 : src, 224 : src*2}

arcface_src = np.array([
  [33.2946, 33.6963],
  [78.5318, 33.5014],
  [56.0252, 53.7366],
  [36.5493, 74.3655],
  [75.7299, 74.2041] ], dtype=np.float32 )

arcface_src = np.expand_dims(arcface_src, axis=0)

# In[66]:

# lmk is prediction; src is template
def estimate_norm(lmk, image_size = 112, mode='arcface'):
  assert lmk.shape==(5,2)
  tform = trans.SimilarityTransform()
  lmk_tran = np.insert(lmk, 2, values=np.ones(5), axis=1)
  min_M = []
  min_index = []
  min_error = float('inf')

  src = arcface_src/112*image_size

  for i in np.arange(src.shape[0]):
    tform.estimate(lmk, src[i])
    M = tform.params[0:2,:]
    results = np.dot(M, lmk_tran.T)
    results = results.T
    error = np.sum(np.sqrt(np.sum((results - src[i]) ** 2,axis=1)))
#         print(error)
    if error< min_error:
        min_error = error
        min_M = M
        min_index = i
  return min_M, min_index

def norm_crop(img, landmark, image_size=160, mode='arcface'):

  left_eye=[landmark[36][0],landmark[36][1]]
  right_eye = [landmark[45][0],landmark[45][1]]
  nose=[landmark[33][0],landmark[33][1]]
  left_mouth=[landmark[48][0],landmark[48][1]]
  right_mouth=[landmark[54][0],landmark[54][1]]


  landmark_five=np.array([left_eye,right_eye,nose,left_mouth,right_mouth])

  for landmarks_index in range(len(landmark_five)):
      x_y = landmark_five[landmarks_index]
      cv2.circle(img, (int(x_y[0]), int(x_y[1])), 3,
                 (222, 222, 222), -1)


  M, pose_index = estimate_norm(landmark_five, image_size, mode)

  warped = cv2.warpAffine(img, M, (image_size, image_size), borderValue=127.0)

  full_M=np.row_stack((M,np.asarray([0,0,1])))
  
  ###make the label as 3xN matrix
  label = landmark.T
  full_label = np.row_stack((label, np.ones(shape=(1, label.shape[1]))))
  label_rotated = np.dot(full_M, full_label)
  label_rotated = label_rotated[0:2, :]
  landmark=label_rotated.T

  for landmarks_index in range(len(landmark)):
      x_y = landmark[landmarks_index]
      cv2.circle(warped, (int(x_y[0]), int(x_y[1])), 3,
                 (100, 222, 222), -1)

  cv2.namedWindow('crop_warp',0)
  cv2.imshow('crop_warp',warped)
  return
  #return warped