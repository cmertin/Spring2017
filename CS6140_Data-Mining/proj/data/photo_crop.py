from __future__ import print_function, division
import dlib
from PIL import Image
from skimage import io
import matplotlib.pyplot as plt
import sys
import numpy as np

def detect_faces(image):
    # Create a face detector
    face_detector = dlib.get_frontal_face_detector()

    # Run detector and get bounding boxes of the faces on image.
    detected_faces = face_detector(image, 1)
    face_frames = [(int(x.left()-x.left()*.15), int(x.top()-x.top()*.15),
                    int(x.right() + x.right()*.15), int(x.bottom()+x.bottom()*.15)) for x in detected_faces]
    return face_frames

# Load image
img_path = sys.argv[1]
img_out_dir = sys.argv[2]
img_out = img_path[:-4]
img_out = img_out.split('/')
img_out = img_out_dir + img_out[1]
ext = img_path[-4:]

image = io.imread(img_path)

# Detect faces
detected_faces = detect_faces(image)

# Crop faces and plot
for n, face_rect in enumerate(detected_faces):
    img_out_ = img_out + "_" + str(n + 1) + ext
    face = Image.fromarray(image).crop(face_rect)
    w, h = face.size
    plt.clf()
    fig = plt.figure(frameon = False)
    inches = 3
    w_out = inches
    h_out = h/w * w_out
    fig.set_size_inches(w_out, h_out, forward=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(face, aspect="auto")
    plt.savefig(img_out_)
    plt.close()
    
