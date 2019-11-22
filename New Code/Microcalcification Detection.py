#!/usr/bin/env python
# coding: utf-8

# <p>
# Title: Microcalcification Detection <br>
# Author: Parth Sachdev
# 
# This code tries to extract features from a given x-ray image
# of breast. The output are several images that will find 
# microcalcifications from the given input image.
# </p>
# 
# <p>
#     Some important methods:
#     <ul>
#         <li> cv2.cvtColor() method is used to convert an image from one color space to another </li>
#         <li> cv2.convertScaleAbs(): Scales, calculates absolute values, and converts the result to 8-bit </li>
#     </ul>
# </p>

# In[1]:


# Import Libraries
import cv2
import numpy as np
import os


# In[2]:


OUTPUT = "Output/"
if not os.path.exists(OUTPUT):
    os.mkdir(OUTPUT)


# In[3]:


def view_and_save(title, image):
    cv2.imwrite(OUTPUT+title+".png", image)
    cv2.imshow(title, image)
    cv2.waitKey(0)


# In[4]:


# Load the image
file = "Dataset/mdb001c.png"
img = cv2.imread(file)


# In[5]:


# View Original Image
view_and_save("Original Image", img)


# <h3> Image Preprocesing </h3>

# In[6]:


# The Median blur operation is similar to the other averaging methods. 
# Here, the central element of the image is replaced by the median of all the pixels in the kernel area.
# This operation processes the edges while removing the noise.
median_img = cv2.medianBlur(img, 3)
view_and_save("After Median Filtering", median_img)


# In[7]:


# Histogram equalization (https://www.geeksforgeeks.org/histograms-equalization-opencv/)
# is a method in image processing of contrast adjustment using the image’s histogram.
img_to_yuv = cv2.cvtColor(median_img, cv2.COLOR_BGR2YUV)
img_to_yuv[:,:,0] = cv2.equalizeHist(img_to_yuv[:,:,0])
hist_equalization_result = cv2.cvtColor(img_to_yuv, cv2.COLOR_YUV2BGR)

view_and_save("Histogram Equalization", hist_equalization_result)


# In[8]:


# Modifying the histogram image
psy = 0.8
modified_histogram = (median_img + psy*hist_equalization_result)/(1+psy)
view_and_save("Modified histogram image", modified_histogram)


# In[9]:


# Not sure what we are doing here
scaled_abs = cv2.convertScaleAbs(modified_histogram)
gaussian_blur = cv2.GaussianBlur(scaled_abs, (9,9), 10.0)
unsharp_image = cv2.addWeighted(img, 1.5, gaussian_blur, -0.5, 0, scaled_abs)
view_and_save("After lce", unsharp_image)


# In[10]:


# Image Dilution
kernel = np.ones((15,15), np.uint8)
img_dilation = cv2.dilate(unsharp_image, kernel, iterations=1)
view_and_save('Image Dilation', img_dilation)


# <h3> Top-hat Transformation </h3>

# In[11]:


# In mathematical morphology and digital image processing, 
# top-hat transform is an operation that extracts small elements and details from given images.
kernel = np.ones((10,15),np.uint8)
tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)

view_and_save("Top-hat Transformation", tophat)


# <h3> Thresholded Image </h3>

# In[12]:


img = tophat

hist = cv2.calcHist([img],[0],None,[256],[0,256])
hist_norm = hist.ravel() / hist.max()
Q = hist_norm.cumsum()
bins = np.arange(256)
fn_min = np.inf
thresh = -1
for i in range(1,256):
    p1, p2 = np.hsplit(hist_norm,[i]) # probabilities
    q1, q2 = Q[i],Q[255]-Q[i] # cum sum of classes
    if q1 == 0:
        q1 = 0.00000001
        if q2 == 0:
            q2 = 0.00000001
        b1, b2 = np.hsplit(bins,[i]) # weights
        # finding means and variances
        m1, m2 = np.sum(p1*b1)/q1, np.sum(p2*b2)/q2
        v1, v2 = np.sum(((b1-m1)**2)*p1)/q1,np.sum(((b2-m2)**2)*p2)/q2
        # calculates the minimization function
        fn = v1*q1 + v2*q2
        if fn < fn_min:
            fn_min = fn
            thresh = i
ret2, thresh = cv2.threshold(img,thresh,255,cv2.THRESH_BINARY)

view_and_save("Thresholded Image", thresh)


# <h3> Clustered Image </h3>

# In[13]:


img = thresh
Z = img.reshape((-1,3))
Z = np.float32(Z)
# define criteria, number of clusters(K) and apply kmeans()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 8
ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
# Now convert back into uint8, and make original image
center = np.uint8(center)
res = center[label.flatten()]
res2 = res.reshape((img.shape))

view_and_save('Clustered Image', res2)


# <h3> Feature Extraction </h3>

# In[ ]:





# In[14]:


cv2.destroyAllWindows()


# In[ ]:




