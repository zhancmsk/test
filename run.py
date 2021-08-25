import os
import os.path as osp
import shutil
import sys
import argparse
import logging
import datetime
import json
from multiprocessing import Pool, Manager
from typing import Dict, Optional

from utils.annotation_io import *
from utils.dicom_io import *
from utils.database import PatientDatabase

annotation_zips = {
    # organ segmentation
    'liver.zip', 'spleen.zip',
    # liver segment segmentation
    '1.zip', '2.zip', '3.zip', '4.zip', '5.zip', '6.zip', '7.zip', '8.zip',
    # vessel segmentation
    'ivc.zip',  # 下腔静脉
    'pv.zip',  # 门静脉
    'hv.zip',  # 肝静脉
    'yw.zip',  # 疑问的血管
    'nb.zip',  # 脑补的血管
    # lesion segmentation
    'bz.zip',  # 病灶
    'bzyw.zip',  #疑问病灶
    'fqbz.zip',  # 废弃病灶
    'fqbzyw.zip'  #废弃疑问病灶
}

def get_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Pipeline to rearange data downloaded '
                    'from middle platform')
    parser.add_argument('--data-dir', help='Directory of input data.')
    parser.add_argument('--tmp-dir', default='/Jupiter/tmp/rearange',
                        help='Directory for temporary files.')
    parser.add_argument('--output-dir', help='Directory to output.')
    parser.add_argument('--cpus', type=int, default=8,
                        help='Number of CPU cores.')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existed files all the time.')
    args = parser.parse_args()

    if not osp.isdir(args.data_dir):
        raise NotADirectoryError(
            f'args.data_dir ({args.data_dir}) is not a directory.')

    os.makedirs(args.tmp_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    return args


def get_logger(args: argparse.Namespace) -> logging.Logger:
    logging.basicConfig(
        level=logging.DEBUG,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(
                osp.join(args.output_dir, 'rearange.log'),
                mode='w'
            )])
    logger = logging.getLogger()

    return logger


def unzip_case(
        source_file: str,
        destination_dir: str
):
    case_id = osp.splitext(osp.basename(source_file))[0]
    out_dir = osp.join(destination_dir, case_id)
    if osp.isdir(out_dir):
        shutil.rmtree(out_dir)
    unzip(source_file, destination_dir)

    ann_dir = osp.join(out_dir, 'annotation')
    if osp.isdir(ann_dir):
        for root, _, files in os.walk(ann_dir):
            for file in files:
                if file in annotation_zips:
                    path = osp.join(root, file)
                    unzip(path, osp.join(ann_dir, file.split('.')[0]))
                    os.remove(path)

    return out_dir


def register(
        database: PatientDatabase,
        case_id: str = '',
        slices_dir: str = '',
        mvd: str = '',
) -> Optional[str]:
    local_id = database.get_local_id(case_id)
    if local_id is None:
        if slices_dir == '':    ###todo:为什么用这个参数判断
            return local_id
        for root, _, files in os.walk(slices_dir):
            for file in files:
                dicom = read_dicom(osp.join(root, file))
                if dicom is not None:
                    patient_id = get_patient_id(dicom)
                    series_id = get_series_instance_uid(dicom)
                    study_date = get_study_date(dicom)
                    study_time = get_study_time(dicom)
                    if None in (
                            patient_id,
                            series_id,
                            study_date,
                            study_time
                    ):
                        continue

                    study_datetime = study_date + study_time
                    if series_id.startswith('MVD'):
                        local_id = series_id.split('_')[2]
                        database.insert_local_id(
                            patient_id, study_datetime, local_id)   ### todo:再看一下
                    else:
                        local_id = database.get_local_id_by_meta_info(
                            patient_id, study_datetime, mvd)    ### todo：在看一下
                    return local_id
    return local_id


def preprocess(
        case_id: str,
        case_dir: str,
        local_id: str,
        output_dir: str
):
    dicom_list = read_dicom_list(osp.join(case_dir, 'slices'))
    pixel_array, spacing, case_datetime = parse_dicom_list(
        dicom_list, ['pixel_array', 'spacing', 'acquisition_datetime'])

    seg_out_dir = osp.join(output_dir, local_id, 'segmentation')
    os.makedirs(seg_out_dir, exist_ok=True)
    if pixel_array is not None:
        np.save(
            osp.join(
                seg_out_dir,
                'slices-' + case_id + '_ori_raw.npy'),
            pixel_array)

    ann_dir = osp.join(case_dir, 'annotation')
    if osp.isdir(ann_dir):
        ### save ori organ npy output dir
        organ_path = osp.join(
            seg_out_dir, 'slices-' + case_id + '_ori_organ.npy')
        if not osp.isfile(organ_path):
            organ = read_organ_annotation(ann_dir)
            if organ is not None:
                np.save(organ_path, organ)
        ### save ori vessel npy output dir
        vessel_path = osp.join(
            seg_out_dir, 'slices-' + case_id + '_ori_vessel.npy')
        if not osp.isfile(vessel_path):
            vessel = read_vessel_annotation(ann_dir)
            if vessel is not None:
                np.save(vessel_path, vessel)
        ### save ori lesion npy to output dir
        lesion_path = osp.join(
            seg_out_dir, 'slices-' + case_id + '_ori_lesion.npy')
        if not osp.isfile(lesion_path):
            lesion = read_lesion_annotation(ann_dir)
            if lesion is not None:
                np.save(lesion_path, lesion)

    logger.info(f'{case_id}: annotation stored.')
    return {'spacing': spacing, 'datetime': case_datetime}


def save_slices(
        source: str,
        destination: str
):
    os.makedirs(destination, exist_ok=True)
    for dicom in os.listdir(source):
        os.link(osp.join(source, dicom), osp.join(destination, dicom))

    logger.info(f'Create link from {source} to {destination}.')


if __name__ == '__main__':
    args = get_parser()
    logger = get_logger(args)
    database = PatientDatabase()

    # io_pool = Pool(args.cpus // 3)
    io_pool = Pool(1)  ###debug test
    unzipped_cases = []
    for root, _, files in os.walk(args.data_dir):
        for file in files:
            if file.startswith('DI_'):
                dir = osp.basename(root)
                if '_' in dir:
                    _, mvd, category = dir.split('_')
                else:
                    break
                case_id = osp.splitext(file)[0]     # DI_***
                status = 'not processed' \
                    if args.overwrite \
                    else database.get_case_status(case_id)
                if status == 'unzipped':
                    case_dir = osp.join(args.tmp_dir, case_id)
                    if osp.isdir(case_dir):
                        unzipped_cases.append(
                            (case_id, mvd, category, case_dir))
                        logger.info(f'{case_id}: skip unzipping.')
                        continue
                    else:
                        status = 'not processed'
                        database.update_case_progress(case_id, status=status)
                unzipped_cases.append(
                    (case_id,
                     mvd,
                     category,
                     io_pool.apply_async(
                        unzip_case,
                        args=(osp.join(root, file), args.tmp_dir))))

    # preprocess_pool = Pool(args.cpus - args.cpus // 3)
    preprocess_pool = Pool(1)      ###debug test
    processed_cases = []
    saved_cases = []
    for case_id, mvd, category, case_dir in unzipped_cases:
        if not isinstance(case_dir, str):
            case_dir = case_dir.get()
            database.update_case_progress(case_id, status='unzipped')

        local_id = register(database, case_id=case_id)
        slices_dir = osp.join(case_dir, 'slices')
        if local_id is None:
            local_id = register(database, slices_dir=slices_dir, mvd=mvd)
            if local_id is None:
                logger.error(
                    f'Failed to register {case_id}({category}) in {mvd}.')
                continue
            database.update_case_progress(case_id, local_id=local_id)
        ### save file to output dir
        saved_cases.append((
            case_id,
            io_pool.apply_async(
                save_slices,
                args=(
                    slices_dir,
                    osp.join(
                        args.output_dir, local_id, 'slices', case_id)))))

        ### preprocess
        if osp.isdir(case_dir):
            processed_cases.append((
                local_id,
                case_id,
                category,
                preprocess_pool.apply_async(
                    preprocess,
                    args=(
                        case_id,
                        case_dir,
                        local_id,
                        args.output_dir))))
        else:
            logger.error(f'{slices_dir} is not a directory.')
    ### update case progress
    patient_infos = {}
    for local_id, case_id, category, case_info in processed_cases:
        case_info = case_info.get()
        database.update_case_progress(case_id, status='processed')
        case_info['category'] = category
        seq_properties = patient_infos.setdefault(local_id, {})
        seq_properties['slices/' + case_id] = case_info
    ### write info to seq_properties.json
    for local_id, patient_info in patient_infos.items():
        time2rpath_list = {}
        for rpath, case_info in patient_info.items():
            case_datetime = case_info['datetime']
            rpath_list = time2rpath_list.setdefault(case_datetime, [])
            rpath_list.append(rpath)

        for i, rpath_list in enumerate(time2rpath_list.values()):
            ### todo 看一下什么时候长度会大于1，大于1之后也只是写入文件中，后期会进行什么处理？
            if len(rpath_list) > 1:
                for rpath in rpath_list:
                    patient_info[rpath]['group'] = i

        json.dump(
            patient_info,
            open(
                osp.join(args.output_dir, local_id, 'seq_properties.json'),
                'w'),
            indent=2)

    preprocess_pool.close()
    preprocess_pool.join()

    ### update case progress
    for case_id, save_task in saved_cases:
        save_task.get()
        database.update_case_progress(case_id, status='done')

    ####remove tmp file
    logger.info(
        f'Finish rearanging data from middle platform. '
        f'Removing cache in {args.tmp_dir}')
    for root, dirs, files in os.walk(args.tmp_dir):
        for dir in dirs:
            io_pool.apply_async(shutil.rmtree, args=(osp.join(root, dir), ))
        for file in files:
            io_pool.apply_async(shutil.rmtree, args=(osp.join(root, file), ))

    io_pool.close()
    io_pool.join()
