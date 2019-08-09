import csv
import os
from glob import glob
from os.path import join

import matplotlib.patches as patches
import numpy as np
import xmltodict
from matplotlib import pyplot as plt
from natsort import natsorted
from PIL import Image
from sklearn.model_selection import KFold, train_test_split
from tqdm.auto import tqdm

# sequence suffixes
seqs = ['Inp', 'Out', 'T2f', 'T1p', 'T1a', 'T1v', 'T1d', 'Dw1', 'Dw2']

slice_directory = join('Z:\\hcc_ml', 'Slices', 'Positive')
bbox_directory = join('Z:\\hcc_ml', 'BoundingBoxes')

all_bbox_xmls = natsorted(glob(join(bbox_directory, '*.xml')))

# generate slice file names from bbox filename
def get_slice_paths(bbox_file):
    froot = bbox_file[-11:-4]
    fname = 'S' + froot + '_{}.png'
    return join(slice_directory, fname)


def get_bbox_coords(bbox_file):
    with open(bbox_file) as fd:
        cur_bbox = xmltodict.parse(fd.read())
    tags = ['xmin', 'ymin', 'xmax', 'ymax']
    bbox = cur_bbox['annotation']['object']
    if isinstance(bbox, list):
        coords = [[int(b['bndbox'][t]) for t in tags] for b in bbox]
    else:
        coords = [int(bbox['bndbox'][t]) for t in tags]
    return coords


def checkforobject(bbox_file):
    with open(bbox_file) as fd:
        cur_bbox = xmltodict.parse(fd.read())
    return not 'object' in cur_bbox['annotation'].keys()

# Display an image with bounding box
def DisplayImageWithBbox(slice_file, bbox_coords):
    # display an image
    im = np.array(Image.open(slice_file), dtype=np.uint8)
    im = np.flip(im,axis=0)
    # Create figure and axes
    _, ax = plt.subplots(1)
    # Display the image
    ax.imshow(im)
    # Create rectangle patches
    cur_coords = bbox_coords
    if len(cur_coords) == 4:
        cur_coords = [cur_coords]
    for coords in cur_coords:
        x = coords[0]
        y = coords[1]
        w = coords[2] - x
        h = coords[3] - y
        rect = patches.Rectangle(
            (x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
        # Add the patch to the Axes
        ax.add_patch(rect)
    plt.show()


tqdm.write('Loading all slice files...')
all_slice_files = [get_slice_paths(f) for f in tqdm(all_bbox_xmls)]
tqdm.write('Loading all bounding boxes...')
all_bbox_coords = [get_bbox_coords(f) for f in tqdm(all_bbox_xmls)]

# Display a sample image
ind = 803
DisplayImageWithBbox(all_slice_files[ind].format(seqs[4]),all_bbox_coords[ind])

# set up cross validationseed = 1
rng = np.random.RandomState(seed=1) # pylint: disable=no-member
kf = KFold(n_splits=5,random_state=rng)

# run cross validation loop
for fold, (train_index, val_index) in enumerate(tqdm(kf.split(X))):
    train_slices = all_slice_files[train_index]
    val_slices = all_slice_files[val_index]
    train_bbox = all_bbox_coords[train_index]
    val_bbox = all_bbox_coords[val_index]

    # write to CSV file
    tqdm.write('Writing csv files...')
    with open('hcc_retinadata_train_fold{}.csv'.format(fold), mode='w',newline="\n", encoding="utf-8") as file:
        writer = csv.writer(file, delimiter=',', quotechar='"',
                            quoting=csv.QUOTE_MINIMAL)
        for f, coords in tqdm(zip(train_slices, train_bbox), total=len(train_slices)):
            if isinstance(coords[0], list):
                for c in coords:
                    entry = [f] + c + ['lesion']
                    writer.writerow(entry)
            else:
                entry = [f] + coords + ['lesion']
                writer.writerow(entry)

    with open('hcc_retinadata_val_fold{}.csv'.format(fold), mode='w',newline="\n", encoding="utf-8") as file:
        writer = csv.writer(file, delimiter=',', quotechar='"',
                            quoting=csv.QUOTE_MINIMAL)
        for f, coords in tqdm(zip(val_slices,val_bbox), total=len(val_slices)):
            if isinstance(coords[0], list):
                for c in coords:
                    entry = [f] + c + ['lesion']
                    writer.writerow(entry)
            else:
                entry = [f] + coords + ['lesion']
                writer.writerow(entry)
