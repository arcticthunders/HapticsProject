
# coding: utf-8

# In[32]:


import matplotlib.pyplot as plt
import open3d as o3d
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.image as image
import numpy as np
import math as mt
import scipy as sp
import scipy.ndimage
#queue for showing multiple images
def imagequeue(image,title,cmap):
    plt.figure()
    plt.title(title)
    if cmap!=None:
        plt.imshow(image,cmap=plt.get_cmap(cmap))
    else:
        plt.imshow(image)        
def imageto3d(path):
    images = []
    titles = []
    cmaps = []
    img = image.imread(path)
    b= len(img[0])
    l = len(img)
    print('length:',l)
    print('breath:',b)
    titles.append('input image')
    images.append(img)
    cmaps.append(None)
    #conversion of image to black and white
    scale = [0.2989, 0.5870, 0.1140]
    newimg = [[0 for i in range(b)]for i in range(l)]
    for i in range(l):
        for j in range(b):
            newimg[i][j] = np.dot(scale,img[i][j])
    titles.append('black and white image')
    images.append(newimg)  
    cmaps.append('gray')#cmap is short for colormap 
    #applying the median filter
    #3x3 matrix is slid across the image and the centre value is replaced by the median of all the nine values
    medianImg =[[0 for i in range(b)]for j in range(l)]
    for i in range(1,l-1):
        for j in range(1,b-1):
            medianImg[i][j] = np.median([newimg[i][j],newimg[i][j-1],newimg[i][j+1],newimg[i-1][j],newimg[i-1][j-1],newimg[i-1][j+1],
                                        newimg[i+1][j], newimg[i+1][j-1],newimg[i+1][j+1]])
    titles.append('image after median noise removal')
    images.append(medianImg)
    cmaps.append('gray') 
    #calcualting the bit length of pixel from maximum value of the pixel intensity
    for i in range(len(titles)):
        imagequeue(images[i],titles[i],cmaps[i])
    n_max = np.amax(medianImg)
    n = 0
    if n_max<=256:
        n=8
    print('maximum pixel value:',n_max)   #n_max<256, its an 8-bit grayscale image
    roughnessVal = [0 for i in range(8)] #1 for each absolute difference of (i,j) from its 8 neighbours
    roughEst = [[roughnessVal for i in range(b)]for j in range(l)]
    roughness = [[0 for i in range(b)]for j in range(l)]
    #order is calcualted to scale the z to a considerable level for decent viewing
    order = (-1*mt.log10(np.max(newimg)/2**n)+2)/2
    for i in range(1,l-1):
        for j in range(1,b-1):
            roughEst[i][j] = [abs(newimg[i][j]-newimg[i][j-1]),abs(newimg[i][j]-newimg[i][j+1]),
                              abs(newimg[i][j]-newimg[i-1][j]),abs(newimg[i][j]-newimg[i-1][j-1]),abs(newimg[i][j]-newimg[i-1][j+1]),
                                        abs(newimg[i][j]-newimg[i+1][j]), abs(newimg[i][j]-newimg[i+1][j-1]),abs(newimg[i][j]-newimg[i+1][j+1])]
            roughness[i][j] = np.sum(roughEst[i][j])/(len(roughnessVal)*(2**n))*10**order
    roughnessEstimate2D = np.sum(roughness)/((l-2)*(b-2))
    #z value estimation from pixel density and scaling into the order of 10s
    depthImg = newimg.copy()  #this corresponds to the depth estimeate for each pixel
    for i in range(l):
        for j in range(b):
            depthImg[i][j] = newimg[i][j]/(2**n)*10**order
    #removing spatial noise in z estiamate using guassian filter
    sigma = [2, 2]
    denoisedDepthImg = sp.ndimage.filters.gaussian_filter(depthImg, sigma, mode='constant',truncate=1)
    x = np.arange(b)
    y = np.arange(l)
    X,Y = np.meshgrid(x,y)
    Z = np.array(denoisedDepthImg)
    fig = plt.figure()
    ax =fig.gca(projection='3d')
    ax.title.set_text('3d plotting via matplotlib with texture mapping')    
    #mapping the image color pixel by pixel to the 3d surface plot
    ax.plot_surface(X,Y,Z, rstride=1, cstride=1,facecolors=img/255)
    #reshaping the rgb data so as to assign it to the point cloud color attribute
    img_arr = img.reshape(l*b,3)
    #xyz is the reshaped set of all 3d points
    xyz = np.zeros((np.size(X), 3))
    xyz[:, 0] = np.reshape(X, -1)
    xyz[:, 1] = np.reshape(Y, -1)
    xyz[:, 2] = np.reshape(Z, -1)
    pcd = o3d.geometry.PointCloud()
    #conversion of 3d points to point cloud
    pcd.points = o3d.utility.Vector3dVector(xyz)
    #estimating the surface normals
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    #color mapping to the point cloud
    pcd.colors = o3d.utility.Vector3dVector(img_arr/255)
    o3d.visualization.draw_geometries([pcd])
    #connecting the 3d points via triangles using the ball pivot algorithm
    distances = pcd.compute_nearest_neighbor_distance()
    avgDistance = np.mean(distances)
    radius  = 2*avgDistance
    bpa_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd,o3d.utility.DoubleVector([radius, radius * 2]))
    #visualising the final 3d rendering
    o3d.visualization.draw_geometries([bpa_mesh])  
    #this is to estimate the haptic probe trajectory from the reconstructed skin surface
    normals = np.asarray(bpa_mesh.triangle_normals)
    triangles = np.asarray(bpa_mesh.triangles)
    points = np.asarray(pcd.points)
    pcd_hpt = o3d.geometry.PointCloud()
    meanHeight = np.mean(points[:,2])
    # diameter of the haptic probe
    d = 0.5
    xyz_hpt = []
    for k in range(len(triangles)):
        t_points = np.asarray([points[j] for j in triangles[k]])
        #estimating the haptic probe trajectory 
        #HPT = RSS + N.d + gaussianError
        #RSS points are considered to be the mean of all the vertices of the triangles that form the mesh
        newPoint = [np.mean(t_points[:,0]),np.mean(t_points[:,1]),np.mean(t_points[:,2])+d*normals[k][2]]
        xyz_hpt.append(newPoint)
    pcd_hpt.points = o3d.utility.Vector3dVector(xyz_hpt)
    xyz_hpt = np.array(xyz_hpt)
    xs = list(set(xyz_hpt[:,0]))
    Ras = []
    for i in range(len(xs)):
        #height array in straight line for same x values
        hptZs = [xyz_hpt[k][2] for k in range(len(xyz_hpt)) if xyz_hpt[k][0]==xs[i]]
        ln = len(hptZs)
        hptMean = np.mean(hptZs)
        Ra = np.mean([np.abs(hptZs[j]-hptMean) for j in range(ln)])
        Ras.append(Ra)
    HapticRoughnessEstimate = np.mean(Ras)
    print('Roughness Estimate from 2D image:',roughnessEstimate2D)
    print('Haptic Roughness Estimate:',HapticRoughnessEstimate)
    o3d.visualization.draw_geometries([pcd_hpt])


# In[33]:


path = 'testImage.jpg'
imageto3d(path)

