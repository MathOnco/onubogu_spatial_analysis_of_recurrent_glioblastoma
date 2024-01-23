from skimage import io, feature, filters, morphology, exposure, measure, draw, restoration, transform
from scipy import ndimage as ndi
from scipy import stats

import os
import pathlib
import numpy as np

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn import metrics, model_selection
from sklearn.cluster import HDBSCAN
from sklearn.model_selection import check_cv
from sklearn.metrics import check_scoring
from sklearn.base import is_classifier


import colour
import shutil
import pandas as pd
import imblearn
from collections import Counter
from PIL import Image
import sys
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectFromModel, RFECV, RFE
from sklearn.model_selection import StratifiedKFold

from captum.attr import FeaturePermutation
from tabgan.sampler import OriginalGenerator, GANGenerator, ForestDiffusionGenerator

import torch
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler

MOSSY = "MOSSY"
SPOTTY = "SPOTTY"
STRINGY = "STRINGY"

CATS = [MOSSY, SPOTTY, STRINGY]
_CAT_IDS = np.argsort(CATS)
CATS = [CATS[i] for i in _CAT_IDS]
CAT_IDS = np.arange(0, len(CATS))

TILE_WH = 99

ncat = len(CATS)
STR2NUM_DICT = {CATS[i]:CAT_IDS[i] for i in range(ncat)}
NUM2STR_DICT = {CAT_IDS[i]:CATS[i] for i in range(ncat)}

# open intervals, witch right value non-inclusive
PURPLE_H_RANGE = [300, 360]
YELLOW_H_RANGE = [50, 200]
BLUE_H_RANGE = [200, 275]

LABEL_RANGE_DICT = {STR2NUM_DICT[MOSSY]: PURPLE_H_RANGE,
                    STR2NUM_DICT[SPOTTY]: YELLOW_H_RANGE,
                    STR2NUM_DICT[STRINGY]: BLUE_H_RANGE
                    }
LABEL_A = 0.3
outline_j = 0.5
outline_c = 0.5
LABEL_RGB_DICT = {k:jch2rgb(np.array([[outline_j, outline_c, np.mean(v)]])) for k, v in LABEL_RANGE_DICT.items()}



X_POS = "Location_Center_X"
Y_POS = "Location_Center_Y"

QC_COL = "id"
QC_X_COL = "x"
QC_Y_COL = "y"
QC_NUM_LBL = "label_id"
QC_STR_LBL = "label"
QC_CASE_COL = "case"
QC_FOV_COL = "fov"
QC_SITE_COL = "site"
#Features
CELL_COLS = [
    'PDGFRA amp',
    'EGFR amp',
    'CDK4 amp',
    'PDGFRA + EGFR amp',
    'CDK4 + PDGFRA amp',
    'CDK4 + EGFR amp',
    'CDK4 + EGFR + PDGFRA amp',
    'Unclassified TME',
    'Hypoxic Unclassified TME',
    'Immune', 'Hypoxic Immune',
    'Endothelial',
    'Hypoxic Endothelial',
    'EC Mimicry Tumor Cell',
    'Hypoxic EC Mimicry Tumor Cell',
    'Tumor Cell',
    'Hypoxic Tumor Cell',
    'TERT C228T wt',
    'TERT C228T mut'
]

X_POS = "Location_Center_X"
Y_POS = "Location_Center_Y"

QC_TILE_WH_COL = "tile_wh"
QC_COL = "id"
QC_X_COL = "x"
QC_Y_COL = "y"
QC_NUM_LBL = "label_id"
QC_STR_LBL = "label"
QC_CASE_COL = "case"
QC_FOV_COL = "fov"
QC_SITE_COL = "site"
#Features
FEATURE_COL = "feature"
CLS_COL = "classifier"
SCORE_COL = "score"
REFLECTANCE_COL = "Reflectance"



def view_as_scatter(img, cspace_name, cspace_fxn=None, channel_1_idx=None, channel_2_idx=None, channel_3_idx=None, log3d=False, cspace_kwargs=None, mask=None, s=3):

    if mask is None:
        img_flat = img.reshape((-1, 3))
    else:
        img_flat = img[mask > 0]


    unique_colors = np.unique(img_flat, axis=0)
    flat_size = unique_colors.shape[0]

    h = 2
    while flat_size%h != 0:
        h += 1

    w = int(flat_size/h)

    rgb_block = np.reshape(unique_colors, (h, w, 3))
    if cspace_fxn is None:
        cspace = rgb_block
    else:
        if cspace_kwargs is not None:
            cspace = cspace_fxn(rgb_block, **cspace_kwargs)
        else:
            cspace = cspace_fxn(rgb_block)
    if channel_2_idx is None:
        a = cspace[:, :, channel_1_idx]
        a = np.unique(a)
        y = np.random.uniform(-0.01, 0.01, size=a.size)

        plt.scatter(a, y, c=unique_colors, s=s)
        plt.xlabel(cspace_name[channel_1_idx])
    elif channel_3_idx is None:

        a = cspace[:, :, channel_1_idx]
        b = cspace[:, :, channel_2_idx]

        a = a.ravel()
        b = b.ravel()
        plt.scatter(a, b, c=unique_colors/255, s=s)
        plt.xlabel(cspace_name[channel_1_idx])
        plt.ylabel(cspace_name[channel_2_idx])


    else:
        a = cspace[:, :, channel_1_idx]
        b = cspace[:, :, channel_2_idx]
        c = cspace[:, :, channel_3_idx]

        a = a.ravel()
        b = b.ravel()
        c = c.ravel()


        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(a, b, c, c=unique_colors, depthshade=False, edgecolor=unique_colors, lw=0)
        plt.xlabel(cspace_name[channel_1_idx])
        plt.ylabel(cspace_name[channel_2_idx])
        ax.set_zlabel(cspace_name[channel_3_idx])
        if log3d:
            plt.title("Log")


def flatten(l):
    return [item for sublist in l for item in sublist]


def rgb2jab(rgb, cspace='CAM16UCS'):
    eps = np.finfo("float").eps
    if np.issubdtype(rgb.dtype, np.integer) and rgb.max() > 1:
        rgb01 = rgb/255.0
    else:
        rgb01 = rgb

    with colour.utilities.suppress_warnings(colour_usage_warnings=True):
        jab = colour.convert(rgb01+eps, 'sRGB', cspace)

    return jab

def rgb2jch(rgb, cspace='CAM16UCS', h_rotation=0):
    jab = rgb2jab(rgb, cspace)
    jch = colour.models.Jab_to_JCh(jab)
    jch[..., 2] += h_rotation

    above_360 = np.where(jch[..., 2] > 360)
    if len(above_360[0]) > 0:
        jch[..., 2][above_360] = jch[..., 2][above_360] - 360

    return jch


def jch2rgb(jch, cspace="CAM16UCS", h_rotation=0):
    eps = np.finfo("float").eps

    c = jch[..., 1]
    h = np.deg2rad(jch[..., 2] - h_rotation)

    a = c*np.cos(h)
    b = c*np.sin(h)

    jab = np.dstack([jch[..., 0], a, b])

    with colour.utilities.suppress_warnings(colour_usage_warnings=True):
        rgb = colour.convert(jab + eps, cspace, 'sRGB')

    rgb = np.clip(rgb, 0, 1)
    rgb = (255*rgb).astype(np.uint8)

    return rgb


def clean_img(img):

    ### Remove flare
    cr, cc = np.array(img.shape)//2
    w = 20
    min_flare_diameter = 10
    ecc_thresh = 0.8 # How circular (0-1, where 0=circle & 1=line)
    center_slice = img[cr-w:cr+w, cc-w:cc+w]
    _, thresh = filters.threshold_multiotsu(center_slice)
    thresh_img = 255*(center_slice > thresh).astype(int)
    label_image = measure.label(thresh_img)
    labeled_regions = measure.regionprops(label_image)
    potential_flares = [r for r in labeled_regions if 2*(r.area/np.pi) >= min_flare_diameter and r.eccentricity < ecc_thresh]
    if len(potential_flares) == 0:
        flare_mask = np.zeros(img.shape, dtype=np.uint8)
    if len(potential_flares) >= 1:
        flare_idx = np.argmax([r.area for r in potential_flares])
        flare_region = potential_flares[flare_idx]


        flare_bbox = np.array(flare_region.bbox)
        circ_pos = draw.disk(flare_region.centroid, radius=np.round(np.max((flare_bbox[2:] - flare_bbox[0:2])/2)).astype(int), shape=thresh_img.shape)
        temp_flare_mask = np.zeros(thresh_img.shape, dtype=np.uint8)
        temp_flare_mask[circ_pos] = 255
        flare_mask = np.zeros(img.shape, dtype=np.uint8)
        flare_mask[cr-w:cr+w, cc-w:cc+w] = temp_flare_mask
        flare_mask = 255*morphology.binary_dilation(flare_mask, morphology.disk(2)).astype(int)
        img = restoration.inpaint.inpaint_biharmonic(img, flare_mask)
        img = exposure.rescale_intensity(img, out_range=np.uint8)

    ### Remove background
    bg = restoration.rolling_ball(img)
    no_bg = img - bg
    no_bg = np.clip(no_bg, 0, 255)

    return no_bg


def get_tile_edges(img_shape_rc, wh):
    img_wh = img_shape_rc[0:2][::-1]
    x_step = np.min([wh, np.floor(img_wh[0]).astype(int)])
    y_step = np.min([wh, np.floor(img_wh[1]).astype(int)])

    x_pos = np.arange(0, img_wh[0], x_step)
    y_pos = np.arange(0, img_wh[1], y_step)

    return x_pos, y_pos


def get_tile_pos(img_shape_rc, wh):
    x_pos, y_pos = get_tile_edges(img_shape_rc, wh)
    tile_bbox_list = np.array([np.array([[x_pos[i], y_pos[j]], [x_pos[i+1], y_pos[j+1]]]) for j in range(len(y_pos) - 1) for i in range(len(x_pos) - 1)])

    return tile_bbox_list


def get_img_id(img_f):
    img_number_filled = [x for x in os.path.split(img_f)[-1].split("_") if x[0].isnumeric()][0]
    img_number_filled = img_number_filled.split(".")[0]
    non_zero_idx = [i for i in range(len(img_number_filled)) if img_number_filled[i] !="0"][0]
    img_id = eval(img_number_filled[non_zero_idx:])

    return img_id


def get_case_id(img_f):
    """
    Get case name that will match cell segementation file
    """
    case_id = os.path.split(img_f)[-1].split("_")[2]
    return case_id


def get_reduced_features(x, labels, feature_names, sample_idx=None, ncv=5, min_features=1):
    """
    """
    if sample_idx is None:
        sample_idx = list(range(x.shape[0]))

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels[sample_idx])
    cv = StratifiedKFold(ncv)
    estimator_list = [RandomForestClassifier(n_estimators=100), ExtraTreesClassifier(n_estimators=100)]
    keep_feature_list = [None] * len(estimator_list)
    keep_feature_names_list = [None] * len(estimator_list)
    acc_list = [None] * len(estimator_list)
    for i, estimator in enumerate(estimator_list):
        if ncv <= 1:
            selector = SelectFromModel(estimator, prefit=False)
        else:
            selector = RFECV(estimator, step=1, cv=cv, min_features_to_select=min_features)

        selector = selector.fit(x[sample_idx, :][0], y)
        feature_idx = selector.get_support(indices=True)
        pred_y = selector.predict(x)
        acc = accuracy_fn(torch.from_numpy(y), torch.from_numpy(pred_y))
        acc_list[i] = acc

        keep_feature_names_list[i] = selector.get_feature_names_out(feature_names)
        keep_feature_list[i] = set(feature_idx)


    consensus_features_idx = list(set.intersection(*keep_feature_list))
    consensus_features_idx = sorted(consensus_features_idx)
    consensus_features = [feature_names[i]  for i in consensus_features_idx]

    return consensus_features, keep_feature_names_list, acc_list


def remove_empty_quadrats(x, y, zero_val=0):
    non_zero_idx = np.where(x.sum(axis=1) > zero_val*x.shape[1])[0]

    print(f"removed {x.shape[0] - len(non_zero_idx)}")
    non_zero_x = x[non_zero_idx, :]
    if y is None:
        non_zero_y = None
    else:
        non_zero_y = y[non_zero_idx]

    return non_zero_x, non_zero_y, non_zero_idx

def calc_cls_weights_for_unbalanced(y):
    """
    See https://medium.com/@zergtant/use-weighted-loss-function-to-solve-imbalanced-data-classification-problems-749237f38b75
    dset: DataLoader
    """

    is_dl = isinstance(y, torch.utils.data.dataloader.DataLoader)
    is_torch = isinstance(y, torch.Tensor)
    if is_dl:
        n_batches = np.ceil(len(y.dataset)/y.batch_size).astype(int)
        batch_y_list = [None] * n_batches
        for i, (batch_x, batch_y) in enumerate(y):

            batch_y_list[i] = batch_y.detach().numpy()

        lbls = np.hstack(batch_y_list)
    elif is_torch:
        lbls = y.detach().numpy()
    else:
        lbls = y

    counts_dict = Counter(lbls)
    counts = np.array(list(counts_dict.values()))[np.argsort(list(counts_dict.keys()))]
    w = np.max(counts)/counts
    w = w/w.sum()

    if is_torch or is_dl:
        w = torch.from_numpy(w.astype(np.float16))

    return w, counts


def numpy2torch_loader(x, y, batch_size=20, shuffle=False, oversample=False):
    dset = TensorDataset(torch.from_numpy(x).type(torch.float), torch.from_numpy(y).type(torch.float))
    if oversample:
        shuffle = False

        class_weights, _ = calc_cls_weights_for_unbalanced(y)

        sample_weights = class_weights[y]

        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(dset), replacement=True)
    else:
        sampler = None

    dataloader = DataLoader(dset, batch_size=batch_size, shuffle=shuffle, sampler=sampler)

    return dataloader


def accuracy_fn(y_true, y_pred):
    """ Calculate accuracy (a classification metric)"""
    correct = torch.eq(y_true, y_pred).sum().item() # torch.eq() calculates where two tensors are equal
    acc = (correct / len(y_pred)) * 100
    return acc


def combine_hypoxic(df):
    hypoxic_cols = [x for x in list(df) if re.search("hypox", x.lower()) is not None]
    col_dict = {x: [y for y in list(df) if y == " ".join(x.split(" ")[1:])][0] for x in hypoxic_cols}
    combined_df = df.copy()
    for x in hypoxic_cols:
        normal_col = col_dict[x]
        combined_df[normal_col] += combined_df[x]

    combined_df = combined_df.drop(columns=hypoxic_cols)

    return combined_df


def get_site(img_f):
    if re.search("recu",img_f.lower()):
        site = "recur"

    elif re.search("prim", img_f.lower()):
        site = "prim"
    else:
        print("can't determine status name")
        site = "unknown"

    return site


def label_regions(img, annotated_img, label_dict):
    """
    un-annoated is `-1`
    """

    if img is None:
        img = np.zeros(annotated_img.shape[0:2])

    region_mask = 255*(np.std(annotated_img, axis=2) != 0)
    labeled_regions = np.full(img.shape[0:2], -1)

    jch_img = rgb2jch(annotated_img)
    for lbl, h_range in label_dict.items():
        lbl_pos = np.where((jch_img[..., 2] >= h_range[0]) &
                           (jch_img[..., 2]  < h_range[1]))

        if len(lbl_pos[0]) == 0:
            continue

        labeled_regions[lbl_pos] = lbl

    labeled_regions[region_mask == 0] = -1

    # Compare to hand annotation
    labeled_rgb = np.dstack([img]*3)
    for lbl, h_range in label_dict.items():
        region_img = 255*(labeled_regions==lbl).astype(np.uint8)
        region_img = ndi.binary_fill_holes(region_img)

        region_img = 255*(region_img).astype(np.uint8)

        if region_img.max() == 0:
            continue

        region_rgb = LABEL_RGB_DICT[lbl]
        labeled_rgb[region_img > 0] = LABEL_A*region_rgb + (1-LABEL_A)*labeled_rgb[region_img > 0]

    view_img = np.hstack([annotated_img, labeled_rgb])

    return labeled_regions, view_img


def label_tile(annotated_tile):
        if annotated_tile.max() < 0:
            return -1
        bc = np.bincount(annotated_tile[annotated_tile >= 0])
        tile_lbl = np.argmax(bc)

        return tile_lbl


def create_labeled_tile_img(labeled_img, tile_wh):
    tile_list = get_tile_pos(labeled_img.shape[0:2], tile_wh)
    tile_x_edges, tile_y_edges = get_tile_edges(labeled_img.shape[0:2], tile_wh)

    nx = len(tile_x_edges) - 1
    ny = len(tile_y_edges) - 1

    labeled_tile_img = np.zeros((ny, nx))

    # tile_list goes across columns, then down rows
    for tile_id, tile_bbox_xy in enumerate(tile_list):
        c0, r0, c1, r1 = tile_bbox_xy.reshape(-1)
        tile = labeled_img[r0:r1, c0:c1]
        tile_label = label_tile(tile)

        tile_r = tile_id // nx
        tile_c = tile_id % nx

        labeled_tile_img[tile_r, tile_c] = tile_label

    return labeled_tile_img


def get_shape_rc_from_file(img_f):
    img = Image.open(str(img_f))
    shape_rc = np.array(img.size[0:2][::-1])
    return shape_rc


def calc_label_freq(y, invert=False):
    c = Counter(y)
    vals = np.array(list(c.values()))[np.argsort(list(c.keys()))]
    n = c.total()
    if invert:
        vals = (n-vals)

    freqs = vals/vals.sum()

    return freqs

def slice_features(x, idx):
    is_df = isinstance(x, pd.DataFrame)
    if is_df:
        sub_x = x.iloc[idx]
    else:
        sub_x = x[idx, ...]

    return sub_x


def get_test_train_split(x, y, min_freq=0.2, max_freq=0.5, test_size=0.1, val_size=0.1, balance_training=True, cat_cols=None, cluster_features=True):
    """
    Want each split to have similar number of each class and similar feature

    """

    Y_IDX = 0
    CLUSTER_IDX = 1
    def remove_singletons(strat_labels):
        n = strat_labels.shape[0]
        cluster_bins = np.arange(strat_labels[:, CLUSTER_IDX].min(), strat_labels[:, CLUSTER_IDX].max() + 2)
        y_bins = np.arange(strat_labels[:, Y_IDX].min(), strat_labels[:, Y_IDX].max()  + 2)

        hist_bins = [None]*2
        hist_bins[Y_IDX] = y_bins
        hist_bins[CLUSTER_IDX] = cluster_bins

        obs_counts, _, _ = np.histogram2d(strat_labels[:, Y_IDX], strat_labels[:, CLUSTER_IDX], bins=hist_bins)
        obs_counts = obs_counts.astype(int)
        singleton_2d = np.where(obs_counts == 1)
        has_singltons = len(singleton_2d) > 0

        if has_singltons:
            singleton_2d = np.dstack(singleton_2d)[0]
            n_singleton = np.sum(obs_counts[singleton_2d[:, 0], singleton_2d[:, 1]])
            singleton_idx = [None] * n_singleton

            for idx, (i, j) in enumerate(singleton_2d):
                s_y_bin = hist_bins[Y_IDX][i]
                s_group_bin = hist_bins[CLUSTER_IDX][j]
                s_idx = np.where((strat_labels[:, Y_IDX]==s_y_bin) &
                                 (strat_labels[:, CLUSTER_IDX]==s_group_bin))[0]

                assert len(s_idx) == 1
                singleton_idx[idx] = s_idx[0]

            sample_idx = [i for i in range(n) if i not in singleton_idx]
        else:
            singleton_idx = []
            sample_idx = list(range(n))

        return sample_idx, singleton_idx


    if cluster_features:
        clusterer = HDBSCAN()
        groups = clusterer.fit_predict(x)
        in_group_idx = np.where(groups >= 0)[0]
        n_groups = np.max(groups[in_group_idx]) + 1
        n_outliers = len(groups[groups < 0])
        print(f"assigned {len(in_group_idx)} samples to {n_groups} groups, and found {n_outliers} outliers")

        labels_2d = np.empty((len(y), 2), dtype=int)
        labels_2d[:, Y_IDX] = y
        labels_2d[:, CLUSTER_IDX] = groups

        sample_idx, singletons = remove_singletons(labels_2d)

        to_split_strat = slice_features(labels_2d, sample_idx)
    else:
        sample_idx = list(range(len(y)))
        to_split_strat = y
        singletons = []

    total_test_size = test_size + val_size

    train_idx, _test_idx = model_selection.train_test_split(sample_idx, test_size=total_test_size, stratify=to_split_strat)

    ## Randomly assign singletons
    r = np.round(np.random.uniform(size=len(singletons))).astype(int)
    for i, s_idx in enumerate(singletons):
        if r[i] == 0:
            train_idx.append(s_idx)
        else:
            _test_idx.append(s_idx)

    x_train = slice_features(x, train_idx)
    y_train = y[train_idx]

    if val_size > 0:
        test_val_ratio = test_size/total_test_size

        if cluster_features:
            _test_labels_2d = slice_features(labels_2d, _test_idx)
            _test_sample_idx, _test_singletons = remove_singletons(_test_labels_2d)
            test_sample_idx = np.array(_test_idx)[_test_sample_idx].tolist()
            test_singletons = np.array(_test_idx)[_test_singletons].tolist()

            test_split_strat = slice_features(labels_2d, test_sample_idx)
        else:
            test_sample_idx = _test_idx
            test_singletons = []
            test_split_strat = y[test_sample_idx]

        test_idx, val_idx = model_selection.train_test_split(test_sample_idx, test_size=test_val_ratio, stratify=test_split_strat)

        r = np.round(np.random.uniform(size=len(test_singletons))).astype(int)
        for i, s_idx in enumerate(test_singletons):
            if r[i] == 0:
                test_idx.append(s_idx)
            else:
                val_idx.append(s_idx)

        x_test= slice_features(x, test_idx)
        y_test = y[test_idx]

        x_val = slice_features(x, val_idx)
        y_val = y[val_idx]

    else:
        test_idx = _test_idx
        x_test= slice_features(x, test_idx)
        y_test = y[test_idx]

        x_val, y_val = None, None

    if not balance_training:
        return x_train, y_train, x_test, y_test, x_val, y_val

    # Generate new examples of rare classes using SMOTE
    o_lbl_counter = Counter(y_train)
    o_counter_keys = list(o_lbl_counter.keys())
    min_target_count = round(o_lbl_counter.total()*min_freq)
    oversampling_strat = {i: min_target_count for i in o_counter_keys if o_lbl_counter[i] < min_target_count}

    if len(oversampling_strat) > 0:
        if cat_cols is None:
            over = imblearn.over_sampling.SMOTE(sampling_strategy=oversampling_strat)
        else:
            over = imblearn.over_sampling.SMOTENC(categorical_features=cat_cols, sampling_strategy=oversampling_strat)
        try:
            sampled_x, sampled_y = over.fit_resample(x_train, y_train)
        except ValueError as e:
            print(f"Can't oversample due to following error: {e}")
            sampled_x, sampled_y = x_train, y_train

    else:
        # Don't need to oversample
        sampled_x, sampled_y = x_train, y_train

    # Undersample classes that are too common
    # u_lbl_counter = Counter(sampled_y)
    # u_counter_keys = list(u_lbl_counter.keys())
    # max_target_count = round(u_lbl_counter.total()*max_freq)
    # undersampling_strat = {i: max_target_count for i in u_counter_keys if u_lbl_counter[i] > max_target_count}
    # if len(undersampling_strat) > 0:
    #     under = imblearn.under_sampling.RandomUnderSampler(sampling_strategy=undersampling_strat)
    #     sampled_x, sampled_y = under.fit_resample(sampled_x, sampled_y)

    return sampled_x, sampled_y, x_test, y_test, x_val, y_val


def gen_data_gan(train_x, train_y, test_x, gen_obj = OriginalGenerator(gen_x_times=1.1)):
    """
    See https://github.com/Diyago/GAN-for-tabular-data

    Results include original observations
    """

    y_is_np = isinstance(train_y, np.ndarray)

    if y_is_np:
        y_df = pd.DataFrame(train_y, columns=list("Y"))
    else:
        y_df = train_y

    new_train, new_target = gen_obj.generate_data_pipe(train_x, y_df, test_x, )

    keep_idx = np.where(np.isnan(new_target) == False)
    new_train = new_train.iloc[keep_idx]
    new_target = new_target.iloc[keep_idx]
    if y_is_np:
        new_target = new_target.values

    return new_train, new_target


def calc_batch_size(n_samples, target_batch_size=50, max_batch_delta=2):
    batch_sizes = np.array([x for x in np.arange(1, n_samples) if
                            (
                            abs(n_samples%x - target_batch_size) <=  max_batch_delta
                            or
                            n_samples%x == 0
                            )])

    target_batches = n_samples/(target_batch_size)

    n_samples/target_batches

    batch_size = int(batch_sizes[np.argmin(np.abs(batch_sizes - target_batch_size))])


    return batch_size
