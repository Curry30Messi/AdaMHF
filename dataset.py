from __future__ import print_function, division
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import torch
from torch.utils.data import Dataset


class WSI_Survival_Dataset(Dataset):
    def __init__(self,
                 csv_file='dataset_csv/ccrcc_clean.csv', data_type='omic', use_signature=False,
                 shuffle_data=False, random_seed=7, num_bins=4, ignored_classes=[],
                 stratify_patients=False, target_column=None, filters={}, epsilon=1e-6, out_of_memory=0):

        self.out_of_memory = out_of_memory
        self.custom_test_ids = None
        self.random_seed = random_seed
        self.stratify_patients = stratify_patients
        self.train_ids, self.validation_ids, self.test_ids = (None, None, None)
        self.data_directory = None

        if shuffle_data:
            np.random.seed(random_seed)
            np.random.shuffle(slide_data)

        print(csv_file)
        slide_data = pd.read_csv(csv_file, low_memory=False)

        if 'case_id' not in slide_data:
            slide_data.index = slide_data.index.str[:12]
            slide_data['case_id'] = slide_data.index
            slide_data = slide_data.reset_index(drop=True)

        if not target_column:
            target_column = 'survival_months'
        else:
            assert target_column in slide_data.columns
        self.target_column = target_column

        if "IDC" in slide_data['oncotree_code']:
            slide_data = slide_data[slide_data['oncotree_code'] == 'IDC']

        unique_patients = slide_data.drop_duplicates(['case_id']).copy()
        uncensored_data = unique_patients[unique_patients['censorship'] < 1]

        discretized_labels, quantile_bins = pd.qcut(uncensored_data[target_column], q=num_bins, retbins=True,
                                                    labels=False)
        quantile_bins[-1] = slide_data[target_column].max() + epsilon
        quantile_bins[0] = slide_data[target_column].min() - epsilon

        discretized_labels, quantile_bins = pd.cut(unique_patients[target_column], bins=quantile_bins, retbins=True,
                                                   labels=False, right=False, include_lowest=True)
        unique_patients.insert(2, 'label', discretized_labels.values.astype(int))

        patient_mapping = {}
        slide_data = slide_data.set_index('case_id')
        for patient in unique_patients['case_id']:
            slide_ids = slide_data.loc[patient, 'slide_id']
            if isinstance(slide_ids, str):
                slide_ids = np.array(slide_ids).reshape(-1)
            else:
                slide_ids = slide_ids.values
            patient_mapping.update({patient: slide_ids})

        self.patient_mapping = patient_mapping

        slide_data = unique_patients
        slide_data.reset_index(drop=True, inplace=True)
        slide_data = slide_data.assign(slide_id=slide_data['case_id'])

        label_mapping = {}
        label_count = 0
        for i in range(len(quantile_bins) - 1):
            for c in [0, 1]:
                label_mapping.update({(i, c): label_count})
                label_count += 1

        self.label_mapping = label_mapping
        for i in slide_data.index:
            key = slide_data.loc[i, 'label']
            slide_data.at[i, 'disc_label'] = key
            censorship = slide_data.loc[i, 'censorship']
            key = (key, int(censorship))
            slide_data.at[i, 'label'] = label_mapping[key]

        self.bins = quantile_bins
        self.num_classes = len(self.label_mapping)
        unique_patients = slide_data.drop_duplicates(['case_id'])
        self.patient_data = {'case_id': unique_patients['case_id'].values, 'label': unique_patients['label'].values}

        reordered_columns = list(slide_data.columns[-2:]) + list(slide_data.columns[:-2])
        slide_data = slide_data[reordered_columns]
        self.slide_data = slide_data
        self.metadata = slide_data.columns[:12]
        self.data_type = data_type
        self.prepare_class_ids()

        self.use_signature = use_signature
        if self.use_signature:
            self.signatures = pd.read_csv('./csv/signatures.csv')
        else:
            self.signatures = None

    def prepare_class_ids(self):
        self.patient_class_ids = [[] for _ in range(self.num_classes)]
        for i in range(self.num_classes):
            self.patient_class_ids[i] = np.where(self.patient_data['label'] == i)[0]

        self.slide_class_ids = [[] for _ in range(self.num_classes)]
        for i in range(self.num_classes):
            self.slide_class_ids[i] = np.where(self.slide_data['label'] == i)[0]

    def prepare_patient_data(self):
        unique_patients = np.unique(np.array(self.slide_data['case_id']))
        patient_labels = []

        for patient in unique_patients:
            locations = self.slide_data[self.slide_data['case_id'] == patient].index.tolist()
            assert len(locations) > 0
            label = self.slide_data['label'][locations[0]]
            patient_labels.append(label)

        self.patient_data = {'case_id': unique_patients, 'label': np.array(patient_labels)}

    @staticmethod
    def prepare_dataframe(data, num_bins, ignored_classes, target_column):
        mask = data[target_column].isin(ignored_classes)
        data = data[~mask]
        data.reset_index(drop=True, inplace=True)
        discretized_labels, bins = pd.cut(data[target_column], bins=num_bins)
        return data, bins

    def __len__(self):
        if self.stratify_patients:
            return len(self.patient_data['case_id'])
        else:
            return len(self.slide_data)

    def get_split_from_dataframe(self, all_splits: dict, split_key: str = 'train', scaler=None):
        split_data = all_splits[split_key]
        split_data = split_data.dropna().reset_index(drop=True)

        if len(split_data) > 0:
            mask = self.slide_data['slide_id'].isin(split_data.tolist())
            filtered_data = self.slide_data[mask].reset_index(drop=True)
            split_data = Generic_Split(filtered_data, metadata=self.metadata, modal=self.data_type,
                                       signatures=self.signatures,
                                       data_dir=self.data_directory, label_col=self.target_column,
                                       patient_dict=self.patient_mapping, num_classes=self.num_classes,
                                       out_of_memory=self.out_of_memory)
        else:
            split_data = None

        return split_data

    def return_splits(self, from_id: bool = True, csv_file: str = None):
        if from_id:
            raise NotImplementedError
        else:
            assert csv_file
            all_splits = pd.read_csv(csv_file)
            train_split = self.get_split_from_dataframe(all_splits=all_splits, split_key='train')
            val_split = self.get_split_from_dataframe(all_splits=all_splits, split_key='val')

            print("****** Normalizing Data ******")
            scalers = train_split.get_scaler()
            train_split.apply_scaler(scalers=scalers)
            val_split.apply_scaler(scalers=scalers)
        return train_split, val_split

    def get_list(self, indices):
        return self.slide_data['slide_id'][indices]

    def get_label(self, indices):
        return self.slide_data['label'][indices]


class MIL_Survival_Dataset(WSI_Survival_Dataset):
    def __init__(self, data_directory, data_type='omic', out_of_memory=0, **kwargs):
        super(MIL_Survival_Dataset, self).__init__(out_of_memory=out_of_memory, **kwargs)
        self.data_directory = data_directory
        self.data_type = data_type
        self.use_h5 = False
        self.out_of_memory = out_of_memory
        if self.out_of_memory > 0:
            print('Using randomly sampled patches [{}] to avoid OOM error'.format(self.out_of_memory))

    def __getitem__(self, index):
        case_id = self.slide_data['case_id'][index]
        label = self.slide_data['disc_label'][index]
        event_time = self.slide_data[self.target_column][index]
        censorship = self.slide_data['censorship'][index]
        slide_ids = self.patient_mapping[case_id]

        if isinstance(self.data_directory, dict):
            source = self.slide_data['oncotree_code'][index]
            data_directory = self.data_directory[source]
        else:
            data_directory = self.data_directory

        if not self.use_h5:
            if self.data_directory:
                if self.data_type == 'path':
                    path_features = []
                    for slide_id in slide_ids:
                        try:
                            wsi_path = os.path.join(data_directory, 'pt_files', '{}.pt'.format(slide_id.rstrip('.svs')))
                            wsi_bag = torch.load(wsi_path)
                            path_features.append(wsi_bag)
                        except FileNotFoundError:
                            continue
                    path_features = torch.cat(path_features, dim=0)
                    if self.out_of_memory > 0 and path_features.size(0) > self.out_of_memory:
                        path_features = path_features[
                            np.random.choice(path_features.size(0), self.out_of_memory, replace=False)]
                    return (path_features, label)
                elif self.data_type == 'omic':
                    omic_path = os.path.join(data_directory, 'omic', '{}.pt'.format(case_id))
                    omic_data = torch.load(omic_path)
                    return (omic_data, label)
        else:
            raise NotImplementedError("HDF5 data access not yet implemented")

        return None

    def get_patient_data(self):
        return self.patient_data
