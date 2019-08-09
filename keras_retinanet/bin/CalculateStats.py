# import keras
import keras

# import keras_retinanet
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color

# import miscellaneous modules
import matplotlib.pyplot as plt
from matplotlib import patches
import cv2
import os
import numpy as np
import time
import csv
from PIL import Image
from natsort import natsorted, index_natsorted, order_by_index
from tqdm.auto import tqdm

# set tf backend to allow memory to grow, instead of claiming everything
import tensorflow as tf

def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True # pylint: disable=no-member
    return tf.Session(config=config)
keras.backend.tensorflow_backend.set_session(get_session())

# paths and parameters
val_csv = 'C:\\Users\\jmj136.UWHIS\\Documents\\keras-retinanet\\keras_retinanet\\hcc_retinadata_val.csv'
model_path = os.path.join('C:\\Users','jmj136.UWHIS','Documents','keras-retinanet',
                          'keras_retinanet','snapshots', 'resnet50_csv_45.h5')
image_output_dir = os.path.join('C:\\Users','jmj136.UWHIS','Documents','keras-retinanet','OutputImages')
seq_list = ['Inp','Out','T2f','T1p','T1a','T1v','T1d','Dw1','Dw2']
seqs = seq_list[3:6]
fold = 0

# load retinanet model
model = models.load_model(model_path, backbone_name='resnet50')

model = models.convert_model(model)

# load label to names mapping for visualization purposes
labels_to_names = {0: 'lesion'}

# Load data to test
def load_image(image_path,seqs):
    image_paths = [image_path.format(s) for s in seqs]
    image_arrays = [np.asarray(Image.open(path).convert('L')) for path in image_paths]
    image = np.stack(image_arrays,axis=-1)
    image = np.flip(image,0)
    return image

with open(val_csv,'r') as f:
    reader = csv.reader(f,delimiter=',')
    data = [row for row in reader]
file_paths = [d[0] for d in data]
all_true_coords = [(int(d[1]), int(d[2]),int(d[3]),int(d[4])) for d in data]
index = index_natsorted(file_paths)
file_paths = order_by_index(file_paths, index)
all_true_coords = order_by_index(all_true_coords, index)
unq_file_paths = natsorted(list(set(file_paths)))


# Functions for processing bounding boxes
from shapely.geometry import Polygon
from itertools import combinations
def GetPolyFromCoords(coords):
    x1 = coords[0]
    y1 = coords[1]
    x2 = coords[2]
    y2 = coords[3]
    return Polygon([(x1,y1),(x2,y1),(x2,y2),(x1,y2)])
def GetOverlapPercent(poly1,poly2):
    return poly1.intersection(poly2).area/poly1.area

# Merge boxes together if overlap above threshold
def MergePolys(in_polys):
    polys = in_polys.copy()
    thresh = .05
    all_done = False
    while not all_done:
        poly_combos = combinations(polys,2)
        overlap_areas = [GetOverlapPercent(p[0],p[1]) for p in poly_combos]
        combo_inds = list(combinations(np.arange(len(polys)),2))
        for c in range(len(combo_inds)):
            if overlap_areas[c] > thresh:
                cur_combo = combo_inds[c]
                new_poly = polys[cur_combo[0]].union(polys[cur_combo[1]]).envelope
                del polys[cur_combo[1]]
                del polys[cur_combo[0]]
                polys.append(new_poly)
                if len(polys) == 1:
                    all_done = True
                break
            elif c == len(combo_inds)-1:
                all_done = True
                break
    return polys

def DisplayImageWithBboxes(image,true_coords=None,pred_coords=None,savepath=None):
    # Create figure and axes
    _, ax = plt.subplots(1,figsize=(15,15))
    # Display the image
    ax.imshow(image,cmap='gray')
    # loop over predicted boxes and make patches
    if pred_coords is not None:
        for coords in pred_coords:
            x = coords[0]
            y = coords[1]
            w = coords[2] - x
            h = coords[3] - y
            rect = patches.Rectangle(
                (x, y), w, h, linewidth=2, edgecolor='r', facecolor='none')
            # Add the patch to the Axes
            ax.add_patch(rect)
    # loop over true boxes and make patches
    if true_coords is not None:
        for coords in true_coords:
            x = coords[0]
            y = coords[1]
            w = coords[2] - x
            h = coords[3] - y
            rect = patches.Rectangle(
                (x, y), w, h, linewidth=2, edgecolor='b', facecolor='none')
            # Add the patch to the Axes
            ax.add_patch(rect)
    if savepath is None:
        plt.show()
    else:
        ax.set_axis_off()
        plt.savefig(savepath, bbox_inches = 'tight', pad_inches = 0)
        plt.close()

# Calculate stats
def CalculateStats(image,true_coords,bthresh=.5,othresh=.5,save=False,savepath=None):
    if save:
        draw = np.copy(image)
        draw = draw[...,1]
    # preprocess
    image = preprocess_image(image)
    image,scale = resize_image(image)
    # process image
    boxes, scores, _ = model.predict_on_batch(np.expand_dims(image, axis=0))
    # correct for image scale
    boxes /= scale
    # Get ground truth polygons
    true_polys = [GetPolyFromCoords(c) for c in true_coords]
    # Get predicted polygons above threshold
    pred_polys = [GetPolyFromCoords(coords) for coords,score in zip(boxes[0], scores[0]) if score>bthresh]
    # if no predicted polygons, then *abort mission*
    if len(pred_polys) == 0:
        tp = 0
        fp = 0
        fn = len(true_coords) 
        tot = len(true_coords)
        # save image, if selected
        if savepath is not None:
            pred_coords = []
            DisplayImageWithBboxes(draw,true_coords,None,savepath=savepath)
            
        return tp,fp,fn,tot
    
    # Merge predicted polygons if needed
    if len(pred_polys)>1:
        clean_polys = MergePolys(pred_polys)
    else:
        clean_polys = pred_polys
    # calculate overlap of each true and predicted polygon
    overlaps = np.zeros((len(clean_polys),len(true_polys)),dtype=np.float)
    for i,true in enumerate(clean_polys):
        for j,pred in enumerate(true_polys):
            overlaps[i,j] = GetOverlapPercent(true,pred)
    # check for overlaps
    true_overlaps = np.max(overlaps,axis=0) > othresh
    pred_overlaps = np.max(overlaps,axis=1) > othresh

    # count true positives
    tp = np.sum(true_overlaps)
    # count false positives
    fp = np.sum(~pred_overlaps)
    # count false negatives
    fn = np.sum(~true_overlaps)
    # total true boxes
    tot = len(true_coords)
    
    # save image, if selected
    if savepath is not None:
        pred_coords = [p.bounds for p in clean_polys]
        DisplayImageWithBboxes(draw,true_coords,pred_coords,savepath=savepath)
    
    return tp,fp,fn,tot

# Run stats calculation across all validation images
true_pos = []
false_pos = []
false_neg = []
total = []
for image_path in tqdm(unq_file_paths):
    # load image
    image = load_image(image_path,seqs)
    # get ground truth coords
    bbox_inds = np.where([fp==image_path for fp in file_paths])[0]
    cur_true_coords = [all_true_coords[i] for i in bbox_inds]
    # calculate stats
    outpath = os.path.join(image_output_dir,image_path[-15:-7])
    # tp,fp,fn,tot = CalculateStats(image,cur_true_coords,bthresh=.45, othresh=.5, save=True, savepath=outpath)
    tp,fp,fn,tot = CalculateStats(image,cur_true_coords,bthresh=.45, othresh=.5, save=True)
    true_pos.append(tp)
    false_pos.append(fp)
    false_neg.append(fn)
    total.append(tot)

# add up stats and display results
tp = np.sum(true_pos)
fp = np.sum(false_pos)
fn = np.sum(false_neg)
total = np.sum(total)

print('----------------------')
print('Classification Results')
print('----------------------')
print('True positives: {}'.format(tp))
print('False positives: {}'.format(fp))
print('False negatives: {}'.format(fn))
print('% Sensitivity: {:.02f}'.format(100*(tp)/(tp+fn)))
print('False positives per image: {:.02f}'.format(fp/len(false_pos)))
print('-----------------------')

# write results to file
tfile = "Results_fold{}.txt".format(fold)
with open(tfile, "w") as text_file:
    text_file.write('True positives: {}\n'.format(tp))
    text_file.write('False positives: {}\n'.format(fp))
    text_file.write('False negatives: {}\n'.format(fn))
    text_file.write('% Sensitivity: {:.02f}\n'.format(100*(tp)/(tp+fn)))
    text_file.write('False positives per image: {:.02f}'.format(fp/len(false_pos)))

print('Results written to {}'.format(tfile))