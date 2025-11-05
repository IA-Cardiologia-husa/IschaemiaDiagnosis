import numpy as np
import pandas as pd
import nibabel as nib
import pydicom
import os
from scipy.ndimage import zoom


def resample(volume, original_spacing, target_spacing=(0.65, 0.65, 0.65)):
    zoom_factors = [o / t for o, t in zip(original_spacing, target_spacing)]
    return zoom(volume, zoom=zoom_factors, order=1)

def normalize_hu(volume, hu_window=(-1000, 1000)):
    vol_clip = np.clip(volume, *hu_window)
    return (vol_clip - hu_window[0]) / (hu_window[1] - hu_window[0])

def crop_or_pad(volume, target_shape=(256, 256, 192)):
    result = np.zeros(target_shape, dtype=volume.dtype)
    input_shape = volume.shape
    crop = [max(0, (i - t) // 2) for i, t in zip(input_shape, target_shape)]
    pad = [max(0, (t - i) // 2) for i, t in zip(input_shape, target_shape)]
    cropped = volume[
        crop[0]:crop[0]+min(input_shape[0], target_shape[0]),
        crop[1]:crop[1]+min(input_shape[1], target_shape[1]),
        crop[2]:crop[2]+min(input_shape[2], target_shape[2])
    ]
    result[
        pad[0]:pad[0]+cropped.shape[0],
        pad[1]:pad[1]+cropped.shape[1],
        pad[2]:pad[2]+cropped.shape[2]
    ] = cropped
    return result

def preprocess(vol, spacing):
    vol = resample(vol, spacing)
    vol = normalize_hu(vol)
    vol = crop_or_pad(vol)
    return vol[np.newaxis,...]

def read_ct(dirpath):
    listfiles = os.listdir(dirpath)
    max_slice = len(listfiles)
    dcmdata = pydicom.dcmread(os.path.join(dirpath, listfiles[0]))

    rows = int(dcmdata[0x0028,0x0010].value)
    columns = int(dcmdata[0x0028,0x0010].value)
    channels = int(dcmdata[0x0028,0x0002].value)

    x_spacing = float(dcmdata[(0x0028, 0x0030)].value[0])
    y_spacing = float(dcmdata[(0x0028, 0x0030)].value[1])
    slice_thickness = float(dcmdata[(0x0018, 0x0050)].value)
    x_origin= float(dcmdata[(0x0020, 0x0032)].value[0])
    y_origin= float(dcmdata[(0x0020, 0x0032)].value[1])

    try:
        rescale_intercept = float(dcmdata[(0x0028, 0x1052)].value)
        rescale_slope = float(dcmdata[(0x0028, 0x1053)].value)
    except:
        rescale_intercept = 0.
        rescale_slope = 1.

    try:
        patient_id = str(dcmdata[(0x0010, 0x0020)].value)
        date = str(dcmdata[(0x0008, 0x0020)].value)
    except:
        patient_id = pd.NA
        date = pd.NaT

    ct_array = np.zeros([rows, columns,max_slice], dtype=np.int32)

    slice_number_list = []
    slice_location_list = []
    for f in listfiles:
        dcmdata = pydicom.dcmread(os.path.join(dirpath, f))
        slice_number = int(dcmdata[(0x0020, 0x0013)].value)
        slice_location = dcmdata[(0x0020, 0x1041)].value
        slice_number_list.append(slice_number)
        slice_location_list.append(slice_location)
        ct_slice = dcmdata.pixel_array
        ct_array[:,:,slice_number-1] = ct_slice[:,:]

    slice_number_list = np.array(slice_number_list)
    slice_location_list = np.array(slice_location_list)

    z_spacing = (slice_location_list.max()-slice_location_list.min())/(slice_number_list.max()-slice_number_list.min())
    z_origin = -slice_location_list.max()
    
    ct_array = ct_array*rescale_slope+rescale_intercept

    return ct_array, (x_spacing, y_spacing, z_spacing),(x_origin, y_origin, z_origin), patient_id, date

def read_nib_mask(filepath):
    img = nib.load(filepath)
    mask = np.array(img.dataobj).transpose([1,0,2])[::-1,:,::-1].astype(bool)
    return mask, img.affine
