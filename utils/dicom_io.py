import os
import os.path as osp
from typing import List, Generator, Optional, Union

import pydicom
import numpy as np


def dicom_generator(path: str) -> Generator:
    for root, _, files in os.walk(path):
        for file in files:
            yield osp.join(root, file)


def read_dicom(path: str) -> Optional[pydicom.dataset.FileDataset]:
    try:
        dicom = pydicom.read_file(path, force=True)
    except Exception:
        dicom = None

    return dicom


def read_dicom_list(path: str) -> List[pydicom.dataset.FileDataset]:
    dicom_list = []
    for dicom_path in dicom_generator(path):
        dicom = read_dicom(dicom_path)
        if dicom is not None:
            dicom_list.append(dicom)

    return dicom_list


def parse_dicom_list(
        dicom_list: List[pydicom.dataset.FileDataset],
        key_list: List[str]
):
    dicom_list.sort(key=get_slice_location)
    results = []
    for key in key_list:
        if key == 'pixel_array':
            results.append(np.stack(
                [get_pixel_array(dicom) for dicom in dicom_list],
                axis=0))
        elif key == 'spacing':
            spacing_2d = list(map(
                float, dicom_list[0]['PixelSpacing'].value))
            if len(dicom_list) == 1:
                thickness = 0.
            else:
                z_range = dicom_list[-1]['ImagePositionPatient'].value[2] - \
                          dicom_list[0]['ImagePositionPatient'].value[2]
                thickness = z_range / (len(dicom_list) - 1)
            results.append([thickness] + spacing_2d)
        elif key == 'series_instance_uid':
            results.append(get_series_instance_uid(dicom_list[0]))
        elif key == 'acquisition_datetime':
            results.append(
                get_acquisition_date(dicom_list[0]) +
                get_acquisition_time(dicom_list[0]))
        else:
            raise NotImplementedError

    return results


def get_patient_id(dicom: pydicom.dataset.FileDataset) -> Optional[str]:
    patient_id = dicom.get((0x0010, 0x0020), None)
    return None if patient_id is None else str(patient_id.value)


def get_study_instance_uid(dicom: pydicom.dataset.FileDataset) -> Optional[str]:
    study_instance_uid = dicom.get((0x0020, 0x000D), None)
    return None if study_instance_uid is None else str(study_instance_uid.value)


def get_study_id(dicom: pydicom.dataset.FileDataset) -> Optional[str]:
    study_id = dicom.get((0x0020, 0x0010), None)
    return None if study_id is None else str(study_id.value)


def get_series_instance_uid(
        dicom: pydicom.dataset.FileDataset
) -> Optional[str]:
    series_instance_uid = dicom.get((0x0020, 0x000E), None)
    return None if series_instance_uid is None else series_instance_uid.value


def get_manufacturer(dicom: pydicom.dataset.FileDataset) -> Optional[str]:
    manufacturer = dicom.get((0x0008, 0x0070), None)
    return None if manufacturer is None else manufacturer.value.split()[0]


def get_series_description(dicom: pydicom.dataset.FileDataset) -> Optional[str]:
    description = dicom.get((0x0008, 0x103E), None)
    return None if description is None else str(description.value)


def get_series_number(dicom: pydicom.dataset.FileDataset) -> Optional[str]:
    series_number = dicom.get((0x0020, 0x0011), None)
    return None if series_number is None else str(series_number.value)


def get_scanning_sequence(
        dicom: pydicom.dataset.FileDataset
) -> Optional[Union[str, List]]:
    scanning_sequence = dicom.get((0x0018, 0x0020), None)
    if scanning_sequence is None:
        return None
    scanning_sequence = scanning_sequence.value
    if isinstance(scanning_sequence, str):
        return scanning_sequence
    elif isinstance(scanning_sequence, pydicom.dataelem.MultiValue):
        return list(scanning_sequence)
    else:
        return None


def get_study_description(dicom: pydicom.dataset.FileDataset) -> Optional[str]:
    description = dicom.get((0x0008, 0x1030), None)
    return None if description is None else str(description.value)


def get_study_date(dicom: pydicom.dataset.FileDataset) -> Optional[str]:
    study_date = dicom.get((0x0008, 0x0020), None)
    return None if study_date is None else str(study_date.value[:8])


def get_study_time(dicom: pydicom.dataset.FileDataset) -> Optional[str]:
    study_time = dicom.get((0x0008, 0x0030), None)
    return None if study_time is None else str(study_time.value[:6])


def get_acquisition_date(dicom: pydicom.dataset.FileDataset) -> Optional[str]:
    acquisition_date = dicom.get((0x0008, 0x0022), None)
    return None if acquisition_date is None else str(acquisition_date.value[:8])


def get_acquisition_time(dicom: pydicom.dataset.FileDataset) -> Optional[str]:
    acquisition_time = dicom.get((0x0008, 0x0030), None)
    return None if acquisition_time is None else str(acquisition_time.value[:6])


def get_slice_location(dicom: pydicom.dataset.FileDataset) -> Optional[float]:
    slice_location = dicom.get((0x0020, 0x0032), None)
    return None if slice_location is None else float(slice_location.value[2])


def get_pixel_array(dicom: pydicom.dataset.FileDataset) -> Optional[np.ndarray]:
    pixel_array = None
    if hasattr(dicom, 'pixel_array'):
        try:
            pixel_array = dicom.pixel_array
        except TypeError:
            pass
    return pixel_array
