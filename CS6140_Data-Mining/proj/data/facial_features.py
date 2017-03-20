from __future__ import print_function, division
import sys
import os
import dlib
import glob
from skimage import io
import numpy as np

def shape_to_np(shape):
    xy = []

    for i in range(68):
        xy.append((max(shape.part(i).x, 1), max(shape.part(i).y, 1)))
    xy = np.asarray(xy, dtype="int")
    return xy

def shape_to_file(shapes, outfile):
    f = open(outfile, 'w')
    for xy in shapes:
        f.write(str(xy[0]) + ',' + str(xy[1]) + '\n')
    f.close()

predictor_path = sys.argv[1] # shape_predictor_68_landmarks.dat
faces_folder_path = sys.argv[2] # directory of faces location

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
win = dlib.image_window()

for f in glob.glob(os.path.join(faces_folder_path, "*.jpg")):
    print("Processing File: {}".format(f))
    out_file = f.split('/')
    out_dir = out_file[0] + '/'
    out_file = out_file[1][:-4] + '.dat'
    out_file = out_dir + out_file
    img = io.imread(f)

    win.clear_overlay()
    win.set_image(img)

    # Ask the detector to find the bounding boxes of each face.
    # The 1 indicates upsampling image 1 time.
    dets = detector(img, 1)
    n_dets = len(dets)
    print("Number of faces detected: {}".format(n_dets))
    for k,d in enumerate(dets):
        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(k, d.left(), d.top(), d.right(), d.bottom()))

        # Get the landmarks/parts for the face in box d
        shape = predictor(img, d)
        shapes = shape_to_np(shape)
        if n_dets == 1:
            shape_to_file(shapes, out_file)
        print(len(shapes), shapes[0])
        print("Part 0: {}, Part 1: {} ...".format(shape.part(0), shape.part(1)))

        # Draw landmarks on screen win.add_overlay(shape)

    win.add_overlay(dets)
    #dlib.hit_enter_to_continue()
