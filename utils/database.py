import os.path as osp
import json
import datetime
from multiprocessing import managers
from typing import List, Dict, Iterable, Union

from utils.dicom_io import *

datetime_format = '%Y%m%d%H%M%S'


class Study:
    def __init__(
            self,
            study_datetime: Union[str, Iterable[str]],
            local_id: str
    ):
        '''
        A scan study, contain some series
        :param study_datetime: scan time
        :param local_id:
        '''
        if isinstance(study_datetime, str):
            study_datetime = {study_datetime, }
        else:
            study_datetime = set(study_datetime)
        self.datetime_set = study_datetime
        self.local_id = local_id

    @property
    def datetime(self):
        return min(self.datetime_set)

    def __contains__(self, ref_datetime: str):
        self_time = datetime.datetime.strptime(self.datetime, datetime_format)
        ref_datetime = datetime.datetime.strptime(ref_datetime, datetime_format)
        return abs(self_time - ref_datetime) < datetime.timedelta(hours=2)

    def add_datetime(self, study_datetime: str):
        self.datetime_set.add(study_datetime)

    def serialize(self):
        return {
            'study_datetime': sorted(list(self.datetime_set)),
            'local_id': self.local_id}


class Patient:
    '''
    A patient, contain some study. bug one patient can be different Patient.
    one patient's scan is a Patient in a day usually
    '''
    def __init__(self, patient_info: List):
        self.studies = []
        self.time2local_id = {}
        for item in patient_info:
            if isinstance(item, dict):
                study = Study(**item)
            elif isinstance(item, Study):
                study = item
            else:
                raise TypeError
            self.append(study)

    def get_local_ids(self):
        return [study.local_id for study in self.studies]

    def append(self, study: Study):
        self.studies.append(study)
        for study_datetime in study.datetime_set:
            self.time2local_id[study_datetime] = study.local_id

    def serialize(self):
        return [study.serialize() for study in self.studies]


class PatientDatabase:
    '''
    database, contain a lot patient
    '''
    def __init__(self):
        os.makedirs(osp.join('/'.join(__file__.split('/')[:-2]), 'databases'),exist_ok=True)
        self.database_dir = osp.join(
            '/'.join(__file__.split('/')[:-2]), 'databases')
        self.patient_infos_path = osp.join(
            self.database_dir, 'patient_infos.json')
        if osp.isfile(self.patient_infos_path):
            patient_infos = json.load(open(self.patient_infos_path))
        else:
            patient_infos = {}

        self.patients = {}
        self.mvds = {}
        for patient_id, patient_info in patient_infos.items():
            patient = Patient(patient_info)
            self.patients[patient_id] = patient
            for local_id in patient.get_local_ids():
                mvd = 'MVD' + str(int(local_id.split('-')[1]))
                local_id_set = self.mvds.setdefault(mvd, set())
                local_id_set.add(local_id)

        self.case_progress_path = osp.join(
            self.database_dir, 'case_progress.json')
        if osp.isfile(self.case_progress_path):
            self.case_progress = json.load(open(self.case_progress_path))
        else:
            self.case_progress = {}

    def asynchronize(self, manager: managers.SyncManager):
        self.case_progress = manager.dict(self.case_progress)

    def register_local_id(self, mvd: str):
        local_id_set = self.mvds.setdefault(mvd, set())
        mvd_id = int(mvd[3:])
        local_id_prefix = f'PA-{mvd_id:0>2}-'
        i = 0
        while True:
            local_id = local_id_prefix + f'{i:0>4}'
            if local_id in local_id_set:
                i += 1
            else:
                break
        local_id_set.add(local_id)

        return local_id

    def insert_study_to_patient(
            self,
            patient_id: str,
            study_datetime: str,
            mvd: str
    ):
        local_id = self.register_local_id(mvd)
        study = Study(study_datetime, local_id)
        self.patients[patient_id].append(study)
        self.save_patient_infos()

        return local_id

    def register_patient(
            self,
            patient_id: str,
            study_datetime: str,
            mvd: str
    ):
        local_id = self.register_local_id(mvd)
        study = Study(study_datetime, local_id)
        self.patients[patient_id] = Patient([study])
        self.save_patient_infos()

        return local_id

    def get_case_status(self, case_id: str):
        return self.case_progress.get(case_id, {}).get('Status', '')

    def get_local_id(self, case_id: str):
        return self.case_progress.get(case_id, {}).get('LocalID', None)

    def get_local_id_by_meta_info(
            self,
            patient_id: str,
            study_datetime: str,
            mvd: str
    ) -> str:
        patient = self.patients.get(patient_id, None)
        if patient is None:
            return self.register_patient(patient_id, study_datetime, mvd)
        else:
            for study in patient.studies:
                if study_datetime in study:
                    study.add_datetime(study_datetime)
                    self.save_patient_infos()
                    return study.local_id
            return self.insert_study_to_patient(patient_id, study_datetime, mvd)

    def insert_local_id(
            self,
            patient_id: str,
            study_datetime: str,
            local_id: str
    ):
        mvd = 'MVD' + str(int(local_id.split('-')[1]))
        local_id_set = self.mvds.setdefault(mvd, set())
        local_id_set.add(local_id)
        study = Study(study_datetime, local_id)
        self.patients[patient_id] = Patient([study])
        self.save_patient_infos()

    def update_case_progress(
            self,
            case_id: str,
            local_id: str = None,
            status: str = None
    ):
        progress = self.case_progress.setdefault(
            case_id,
            {'LocalID': None,
             'Status': 'not processed',
             'LatestModificationDatetime': str(datetime.datetime.now())})
        if local_id is not None:
            progress['LocalID'] = local_id
        if status is not None:
            progress['Status'] = status

        self.save_case_progress()

    def save_patient_infos(self):
        patients = {
            patient_id: patient.serialize()
            for patient_id, patient in self.patients.items()}
        json.dump(patients, open(self.patient_infos_path, 'w'), indent=2)

    def save_case_progress(self):
        json.dump(
            dict(self.case_progress),
            open(self.case_progress_path, 'w'),
            indent=2)
