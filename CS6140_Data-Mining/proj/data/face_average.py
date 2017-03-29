from __future__ import print_function, division
import os
import cv2
import numpy as np
import math
import glob
import sys
import matplotlib.pyplot as plt
import matplotlib

def Read_Points(path, ext=".dat"):
    ext = "*" + ext
    points = {}
    for f in glob.glob(os.path.join(path, ext)):
        outfile = f[:-4]
        vals = []
        lines = [line.strip('\n') for line in open(f)]
        for line in lines:
            xy = line.split(',')
            vals.append((int(xy[0]), int(xy[1])))
        points[outfile] = np.array(vals)
        
    return points

def Read_Images(path, points, ext=".jpg"):
    ext = "*" + ext
    images = {}

    for f in glob.glob(os.path.join(path, ext)):
        outfile = f[:-4]
        if outfile in points:
            # Read the image
            img = cv2.imread(f)
            
            # Convert to floating point
            img = np.float32(img)/255.0

            # Add to array of images
            images[outfile] = img
    return images

def Similarity_Transform(inPoints, outPoints):
    s60 = math.sin(60 * math.pi/180)
    c60 = math.cos(60 * math.pi/180)

    inPts = np.copy(inPoints).tolist()
    outPts = np.copy(outPoints).tolist()

    xin = c60 * (inPts[0][0] - inPts[1][0]) - s60 * (inPts[0][1] - inPts[1][1]) + inPts[1][0]
    yin = s60 * (inPts[0][0] - inPts[1][0]) + c60 * (inPts[0][1] - inPts[1][1]) + inPts[1][1]

    inPts.append([np.int(xin), np.int(yin)])

    xout = c60 * (outPts[0][0] - outPts[1][0]) - s60 * (outPts[0][1] - outPts[1][1]) + outPts[1][0]
    yout = s60 * (outPts[0][0] - outPts[1][0]) + c60 * (outPts[0][1] - outPts[1][1]) + outPts[1][1]

    outPts.append([np.int(xout), np.int(yout)])

    tform = cv2.estimateRigidTransform(np.array([inPts]), np.array([outPts]), False)

    return tform

def Rect_Contains(rect, point):
    if point[0] < rect[0]:
        return False
    elif point[1] < rect[1]:
        return False
    elif point[0] > rect[2]:
        return False
    elif point[1] > rect[3]:
        return False
    return True

def Calculate_Delaunay_Triangles(rect, points):
    # Create subdivision
    subdiv = cv2.Subdiv2D(rect)

    # Insert points into subdiv
    for p in points:
        subdiv.insert((p[0], p[1]))

    # List of triangles, each is a list of 3 points (6 numbers)
    triangle_list = subdiv.getTriangleList()

    # Find the indices of triangles in the points array
    delaunayTri = []

    for t in triangle_list:
        pt = []
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])
        pt.append(pt1)
        pt.append(pt2)
        pt.append(pt3)

        if Rect_Contains(rect, pt1) and Rect_Contains(rect, pt2) and Rect_Contains(rect, pt3):
            ind = []
            for j in range(3):
                for idx in range(len(points)):
                    if abs(pt[j][0] - points[idx][0]) < 1.0 and abs(pt[j][1] - points[idx][1]) < 1.0:
                        ind.append(idx)
            if len(ind) == 3:
                delaunayTri.append((ind[0], ind[1], ind[2]))
    return delaunayTri

def Constrain_Point(p, w, h):
    return (min(max(p[0], 0), w-1), min(max(p[1],0),h-1))

def Apply_Affine_Transform(src, srcTri, dstTri, size):
    # Given a pair of triangles find the affine transform
    warp_map = cv2.getAffineTransform(np.float32(srcTri), np.float32(dstTri))

    # Apply the affine transform just found to the source image
    dst = cv2.warpAffine(src, warp_map, (size[0], size[1]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

    return dst

# Warps and alpha blends triangular regions from img1 and img2 to img
def Warp_Triangle(img1, img2, t1, t2):

    # Find bounding rectange for each triangle
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))

    # Offset points by left top corner of the respective rectangles
    t1_rect = []
    t2_rect = []
    t2_rect_int = []

    for i in range(3):
        t1_rect.append(((t1[i][0] - r1[0]), (t1[i][1] - r1[1])))
        t2_rect.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))
        t2_rect_int.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))

    # Get mask by filling triangle
    mask = np.zeros((r2[3], r2[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(t2_rect_int), (1.0, 1.0, 1.0), 16, 0)

    # Apply warp_image to small rectangular patches
    img1_rect = img1[r1[1]:r1[1]+r1[3], r1[0]:r1[0]+r1[2]]

    size = (r2[2], r2[3])

    img2_rect = Apply_Affine_Transform(img1_rect, t1_rect, t2_rect, size)

    img2_rect = img2_rect * mask

    # Copy triangular region of the rectangular patch to the output image
    img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] *=  ( (1.0, 1.0, 1.0) - mask )
     
    img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] += img2_rect


city = "Provo"
out_dir = city + "_Affine/"
path = city + "_Crop/"

w = 100
h = 100

points = Read_Points(path)
images = Read_Images(path, points)

eye_corners = [(np.int(0.3 * w), np.int(h/3)), (np.int(0.7 * w), np.int(h/3))]

images_norm = []
points_norm = []

# Add boundary points for delaunay traingulation
boundary_pts = np.array([(0,0), (w/2, 0), (w-1,0), (w-1,h/2), (w-1,h-1), (w/2, h-1), (0, h-1), (0,h/2)])

# Initialize location of average points to be 0's
n = points.keys()
n = len(points[n[0]])
points_avg = np.array([(0,0)] * (n + len(boundary_pts)), np.float32())

# Warp images and transform landmarks to output coordinate system,
# find average of transformation landmarks

for key in sorted(points.iterkeys()):
    p1 = points[key]

    # Corners of the eye in the input image
    eye_corner_src = [p1[36], p1[45]]

    # Compute similarity transform
    tform = Similarity_Transform(eye_corner_src, eye_corners)

    # Apply similarity transform
    img = cv2.warpAffine(images[key], tform, (w,h))

    # apply similarity transform on points
    p2 = np.reshape(np.array(p1), (68,1,2))

    p = cv2.transform(p2, tform)
    p = np.float32(np.reshape(p, (68,2)))

    # Append boundary points, will be used in Delaunay Triangulation
    p = np.append(p, boundary_pts, axis=0)

    # Calculate the location of average landmark points
    points_avg += p/len(points.keys())

    # Output the affine transformed images to a new directory for future
    out_file = out_dir + key.split('/')[1] + ".jpg"
    img_ =  cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    r, g, b = img_[:,:,0], img_[:,:,1], img_[:,:,2]
    img_ = 0.2989 * r + 0.5870 * g + 0.1140 * b
        
    plt.clf()
    fig = plt.figure(frameon = False)
    inches = int(max(w,h)/100)
    w_out = inches
    h_out = h/w * w_out
    fig.set_size_inches(w_out, h_out, forward=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(img_, cmap="gray", aspect="auto")
    plt.savefig(out_file, cmap="gray")
    plt.close()
    
    #cv2.imshow("Image", img)
    #cv2.waitKey(0)
    points_norm.append(p)
    images_norm.append(img)

# Delaunay triangulation
rect = (0, 0, w, h)
dt = Calculate_Delaunay_Triangles(rect, np.array(points_avg))

# Output image
output = np.zeros((h,w,3), np.float32())

# Warp input images to average image landmarks
for i in range(len(images_norm)):
    img = np.zeros((h,w,3), np.float32())

    # Transform triangles one by one
    for j in range(len(dt)):
        tin = []
        tout = []
        for k in range(3):
            pIn = points_norm[i][dt[j][k]]
            pIn = Constrain_Point(pIn, w, h)

            pOut = points_avg[dt[j][k]]
            pOut = Constrain_Point(pOut, w, h)

            tin.append(pIn)
            tout.append(pOut)
        Warp_Triangle(images_norm[i], img, tin, tout)

    # Add image intensities for averaging
    output += img

# Divide by number of images to get average
output = output/len(points.keys()) 
output2 = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
# Display result
out_file = city + "_avg.jpg"

fig = plt.figure(frameon = False)
inches = 6
w_out = inches
h_out = h/w * w_out
fig.set_size_inches(w_out, h_out, forward=False)
ax = plt.Axes(fig, [0., 0., 1., 1.])
ax.set_axis_off()
fig.add_axes(ax)
ax.imshow(output2, aspect="auto")
#plt.savefig(out_file)
plt.close()

cv2.imshow("Image", output)
cv2.waitKey(0)
