import os
import os.path as osp
from zipfile import is_zipfile, ZipFile
import logging
from typing import Optional

import SimpleITK as sitk
import numpy as np

from utils.dicom_io import read_dicom_list, parse_dicom_list


def unzip(source: str, destination: str):
    if is_zipfile(source):
        zf = ZipFile(source)
        zf.extractall(destination)
        zf.close()


def save_numpy_as_niigz(arr: np.ndarray, destination: str):
    sitk_image = sitk.GetImageFromArray(arr)
    sitk.WriteImage(sitk_image, destination)


def read_segmentation(path: str) -> Optional[np.ndarray]:
    dicom_list = read_dicom_list(path)
    if dicom_list:
        segmentation, = parse_dicom_list(dicom_list, ['pixel_array'])
        return segmentation
    else:
        return None


def read_liver_annotation(path: str) -> Optional[np.ndarray]:
    liver = read_segmentation(osp.join(path, 'liver'))
    if liver is not None:
        liver = liver.astype(np.int8)
        liver = np.clip(liver, 0, 1)
        
    return liver


def read_spleen_annotation(path: str) -> Optional[np.ndarray]:
    spleen = read_segmentation(osp.join(path, 'spleen'))
    if spleen is not None:
        spleen = spleen.astype(np.int8)
        spleen = np.clip(spleen, 0, 1)

    return spleen


def read_organ_annotation(path: str) -> Optional[np.ndarray]:
    logger = logging.getLogger(__name__)        ### todo:

    liver = read_liver_annotation(path)
    spleen = read_spleen_annotation(path)
    ### todo: 看一下下面的逻辑
    if liver is None:
        if spleen is None:
            organ = None
        else:
            organ = spleen * 2
    else:
        if spleen is None:
            organ = liver
        else:
            if liver.shape != spleen.shape:
                logger.error(
                    f'{path}: unconsistent shape of liver and '
                    'spleen annotations.')
                return None
            organ = liver + spleen * 2
            organ = np.clip(organ, 0, 2)

    return organ


def read_liver_segments_annotation(path: str) -> Optional[np.ndarray]:
    logger = logging.getLogger(__name__)

    liver_segments = [
        read_segmentation(osp.join(path, str(i)))
        for i in range(1, 9)]
    shape = None
    for liver_segment in liver_segments:
        if liver_segment is not None:
            shape = liver_segment.shape
            break
    if shape is None:
        return None
    for i, liver_segment in enumerate(liver_segments):
        if liver_segment is None:
            liver_segments[i] = np.zeros(shape)
        else:
            if liver_segment.shape != shape:
                logger.error(
                    f'{path}: unconsistent shape of liver segments '
                    'annotations.')
                return None
            liver_segments[i] = np.clip(liver_segment, 0, 1) * (i + 1)
    liver_segments = np.stack(liver_segments, axis=0)
    liver_segments = np.amax(liver_segments, axis=0)
    liver_segments = liver_segments.astype(np.int8)

    return liver_segments


def read_vessel_annotation(path: str) -> Optional[np.ndarray]:
    logger = logging.getLogger(__name__)
    ### todo 看一下逻辑
    vessels = [
        read_segmentation(osp.join(path, vessel))
        for vessel in ['hv', 'pv', 'ivc', 'nb', 'yw']]
    shape = None
    for vessel in vessels:
        if vessel is not None:
            shape = vessel.shape
            break
    if shape is None:
        return None
    for i, vessel in enumerate(vessels):
        if vessel is None:
            vessels[i] = np.zeros(shape)
        else:
            if vessel.shape != shape:
                logger.error(
                    f'{path}: unconsistent shape of vessels annotations.')
            vessels[i] = np.clip(vessel, 0, 1) * (i + 1)
    vessels = np.stack(vessels, axis=0)
    vessels = np.amax(vessels, axis=0)
    vessels = vessels.astype(np.int8)

    return vessels


def read_lesion_annotation(path: str) -> Optional[np.ndarray]:
    logger = logging.getLogger(__name__)

    if not set(os.listdir(path)).isdisjoint(
            ('fqbz', 'fqbzyw', 'fqbz.zip', 'fqbzyw.zip')):
        return None

    lesion_dir = osp.join(path, 'bz')
    if osp.isdir(lesion_dir):
        lesion = read_segmentation(lesion_dir)
    else:
        lesion = None
    qsn_lesion_dir = osp.join(path, 'bzyw')
    if osp.isdir(qsn_lesion_dir):
        qsn_lesion = read_segmentation(osp.join(path, 'bzyw'))
    else:
        qsn_lesion = None

    if lesion is None:
        if qsn_lesion is not None:
            lesion = qsn_lesion.astype(np.int8) * 2
    else:
        lesion = lesion.astype(np.int8)
        if qsn_lesion is not None:
            qsn_lesion = qsn_lesion.astype(np.int8)
            if lesion.shape != qsn_lesion.shape:
                logger.error(
                    f'{path}: unconsistent shape of lesion and '
                    'questioned lesion annotation')
                return None
            lesion += qsn_lesion * 2
            lesion = np.clip(lesion, 0, 2)

    return lesion
