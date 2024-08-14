# Licensed under the MIT License.
# Copyright (c) Microsoft Corporation.

import torch
from numba import njit
from typing import List
from .tool_battery_data import BatteryData
import numpy as np
from scipy.interpolate import interp1d
import abc
from tqdm import tqdm
from .tool_train_test_split import split
from .setting import seed
import os
import pickle
from .directory import train_dict, test_dict
from .tool_databundle import DataBundle
from .tool_transformation import ZScoreDataTransformation, SequentialDataTransformation, LogScaleDataTransformation
from sklearn.linear_model import LinearRegression
import math


#process: get_Qdlin -> _get_Qdlin -> interpolate

#x->V , y->Q
def interpolate(x, y, interp_dims, xs=0, xe=1):
    if len(x) <= 2:
        return np.zeros(interp_dims)
    #data를 먼저 주고 interpolate function을 생성
    func = interp1d(x, y, bounds_error=False, kind='linear', fill_value="extrapolate")
    #원하는 x데이터 범위만큼 array 생성 
    new_x = np.linspace(xs, xe, interp_dims)
    #정한 범위의 x데이터를 생성한 interpolate function에 집어 넣어 데이터 얻음
    return func(new_x)

def _get_Qdlin(I, V, Q, min_V, max_V):  # noqa
    #should check data scale! if mA or mV then it could make mis slacing 
    eps = 1e-3
    I, V, Q = np.array(I), np.array(V), np.array(Q)
    #Use discharging data
    y = interpolate(V[I < -eps], Q[I < -eps], 1000, xs=min_V, xe=max_V)
    return y[::-1]

def get_Qdlin(cell_data, cycle_data, use_precalculated=False):
    #Qdlin(already interpolated discharging Qapacity data)
    if 'Qdlin' in cycle_data.additional_data and use_precalculated:
        return np.array(cycle_data.additional_data['Qdlin'])
    return _get_Qdlin(
        cycle_data.current_in_A,
        cycle_data.voltage_in_V,
        cycle_data.discharge_capacity_in_Ah,
        cell_data.min_voltage_limit_in_V,
        cell_data.max_voltage_limit_in_V)

#---------------------------------------------------------------------------------

@njit
def smooth(x, window_size=10, sigma=3):
    res = np.empty_like(x)
    meds = np.empty_like(x)
    for i in range(len(x)):
        low = max(0, i-window_size)
        high = min(len(x), i+window_size+1)
        meds[i] = np.median(x[low: high])
    base = np.std(np.abs(x - meds))
    for i in range(len(x)):
        if np.abs(meds[i] - x[i]) > base * sigma:
            res[i] = meds[i]
        else:
            res[i] = x[i]
    return meds

@njit
def get_charge_time(I, t):  # noqa
    res = 0.
    for i in range(1, len(I)):
        if I[i] < 0:
            res += t[i] - t[i-1]
    return res

#-------------------------------------------------------------------------------------

class BaseFeatureExtractor(abc.ABC):
    def __call__(self):
        pbar = tqdm(self.train_cells, desc='Extracting features')
        # features = torch.stack([self.process_cell(cell) for cell in pbar])
        features = []
        labels=[]
        for i, cell in enumerate(pbar):
            features.extend(self.process_cell(cell))
            labels.extend(self.label_annotation(cell))
        train_features = torch.stack(features).float()
        train_labels=torch.stack(labels).float().view(-1)
        pickle.dump(train_features.float(), open(os.path.join(train_dict,f'{self.test_name}_train_features.pkl'),'wb'))
        pickle.dump(train_labels, open(os.path.join(train_dict,f'{self.test_name}_train_labels.pkl'),'wb'))

        pbar = tqdm(self.test_cells, desc='Extracting features')
        # features = torch.stack([self.process_cell(cell) for cell in pbar])
        features = []
        labels=[]
        for i, cell in enumerate(pbar):
            features.extend(self.process_cell(cell))
            labels.extend(self.label_annotation(cell))
        test_features = torch.stack(features).float()
        test_labels=torch.stack(labels).float().view(-1)
        pickle.dump(test_features.float(), open(os.path.join(test_dict,f'{self.test_name}_test_features.pkl'),'wb'))
        pickle.dump(test_labels, open(os.path.join(test_dict,f'{self.test_name}_test_labels.pkl'),'wb'))

        dataset = DataBundle(
            train_features, train_labels, test_features, test_labels,
            feature_transformation=self.feature_transformation,
            label_transformation=self.label_transformation
        )
        return dataset

    @abc.abstractmethod
    def process_cell(self, cell_data: BatteryData) -> torch.Tensor:
        """Generate feature for a single cell.
`
        Args:
            cell_data (BatteryData): data for single cell.

        Returns:
            torch.Tensor: the processed feature.
        """

class Make_train_data(BaseFeatureExtractor):
    def __init__(self, SEED=seed, feature_transform=True, cyc_interval: int = 10): 
        self.train_cells, self.test_cells = split()
        self.cyc_interval=cyc_interval
        self.test_name='classify_'+f'seed_{seed}'+f'cyc_interval_{self.cyc_interval}'
        self.feature_transformation=ZScoreDataTransformation()
        if feature_transform==False:
            self.feature_transformation=None
        self.label_transformation=None
        self.smooth_diff_qdlin=True
        self.cyc_interval=cyc_interval
        self.interp_dim = 1000
        self.min_cycle_index = 2
        self.use_precalculated_qdlin = False
        self.diff_base=2


        cycles_to_keep=None
        if cycles_to_keep is not None and isinstance(cycles_to_keep, int):
            cycles_to_keep = [cycles_to_keep]
        self.cycles_to_keep = cycles_to_keep

        self.smooth = True

        # See https://github.com/petermattia/revisit-severson-et-al/blob/main/revisit-severson-et-al.ipynb noqa
        self.cycle_average = None

        self.extracted_cyc_list=[]

    def process_cell(self, cell_data: BatteryData) -> torch.Tensor:        
        # chr_rate=cell_data.charge_protocol[0].rate_in_C

        #min cyc index로 부터의 data 갯수, short circuit cycle은 제외함
        data_len=cell_data.short_circuit_cycle-(self.min_cycle_index+1)+1-1
            
        self.feature_list=['Minimum', 'Variance', 'Skewness', 'Kurtosis', 'Mean', 'Maximum',    
                       'ratio between max discharge capacity and early discharge capacity', 
                       'Slope of linear fit to the capacity curve', 'Intercept of linear fit to the capacity curve',    
                       'Average early charge time']
        features=[]  
        for l in range(data_len):
            one_sample=[]
            self.low_cycle_index = self.min_cycle_index
            self.high_cycle_index = self.min_cycle_index + l 

            #extra_feature-->last cycle에 대한 diff_qdlin에 대한 statistcal feature+full model features에 대한 것
            extra_feature = self.__get_features(cell_data, self.feature_list)
            extra_feature = torch.tensor(extra_feature)
            
            # Reshape extra_feature properly
            extra_feature = extra_feature.reshape(-1)
    
            # Append combined_feature to features
            features.append(extra_feature)
        
        features = torch.stack(features)
        features=features.float()
        
        # Fill NaN
        features[torch.isnan(features) | torch.isinf(features)] = 0.

        self.all_features_list=['early 10 cycle min', 'early 10 cycle var', 'early 10 cycle ske', 'early 10 cycle kur', 'early 10 cycle mean', 'early 10 cycle max',  
                                'ratio between max discharge capacity and early discharge capacity', 'Slope of linear fit to the capacity curve',   
                                'Intercept of linear fit to the capacity curve', 'Average early charge time',
                                'last 10 cycle min', 'last 10 cycle var', 'last 10 cycle ske', 'last 10 cycle kur', 'last 10 cycle mean', 'last 10 cycle max']
        
        return features


    def __get_features(self, cell_data: BatteryData,
                    feature_lists: List) -> torch.Tensor:
        early_cycle = cell_data.cycle_data[self.low_cycle_index]
        after_early_10_cycle = cell_data.cycle_data[self.low_cycle_index+self.cyc_interval-1]
        before_last_10_cycle=cell_data.cycle_data[self.high_cycle_index-self.cyc_interval+1]
        last_cycle=cell_data.cycle_data[self.high_cycle_index]

        self.extracted_cyc_list.append([self.low_cycle_index, self.low_cycle_index+self.cyc_interval-1, self.high_cycle_index-self.cyc_interval+1, self.high_cycle_index, cell_data.short_circuit_cycle])

        early_qdlin = get_Qdlin(
            cell_data, early_cycle, self.use_precalculated_qdlin)
        after_early_10_qdlin = get_Qdlin(
            cell_data, after_early_10_cycle, self.use_precalculated_qdlin)
        before_last_10_qdlin= get_Qdlin(
            cell_data, before_last_10_cycle, self.use_precalculated_qdlin)
        last_qdlin= get_Qdlin(
            cell_data, last_cycle, self.use_precalculated_qdlin)

        #diff_qdlin feature in nat.energy paper
        diff_qdlin = after_early_10_qdlin - early_qdlin
        if self.smooth_diff_qdlin:
            diff_qdlin = smooth(diff_qdlin)
        diff_qdlin = torch.from_numpy(diff_qdlin)
        diff_qdlin = diff_qdlin[~diff_qdlin.isnan()]
        if len(diff_qdlin) <= 1:
            raise ValueError('Qdlin is all nan!')
        
        gap_n_diff_qdlin = last_qdlin - before_last_10_qdlin
        if self.smooth_diff_qdlin:
            gap_n_diff_qdlin = smooth(gap_n_diff_qdlin)
        gap_n_diff_qdlin = torch.from_numpy(gap_n_diff_qdlin)
        gap_n_diff_qdlin = gap_n_diff_qdlin[~gap_n_diff_qdlin.isnan()]
        if len(gap_n_diff_qdlin) <= 1:
            raise ValueError('Qdlin is all nan!')

        results = []
        # for feature in feature_lists:
        #     value = self.__get_feature(cell_data, diff_qdlin, feature)
        #     if value is not None:
        #         results.append(value)
        for feature in feature_lists:
            value = self.__get_feature(cell_data, diff_qdlin, feature)
            if value is not None:
                results.append(value)
        
        for feature in ['Minimum', 'Variance', 'Skewness', 'Kurtosis', 'Mean', 'Maximum']:
            value = self.__get_feature(cell_data, gap_n_diff_qdlin, feature)
            if value is not None:
                results.append(value)

        results = torch.tensor(results)

        # Fill NaN and Inf
        results[torch.isnan(results) | torch.isinf(results)] = 0.

        return results


    def __get_feature(self, cell_data: BatteryData,
                    diff_qdlin: torch.Tensor,
                    feature: str) -> float:
        eps = 1e-8
        # delta Qd features
        Qd_features = {
            'Minimum': lambda x: math.log10(abs(x.min()) + eps), #eps is for being non zero in log or denominator
            'Variance': lambda x: math.log10(abs(x.var()) + eps),
            'Skewness': lambda x: math.log10(
                (abs((x - x.mean()) ** 3).mean()) / (x.std() ** 3 + eps) + eps
            ),
            'Kurtosis': lambda x: math.log10(
                ((x - x.mean()) ** 4).mean() / (x.var() ** 2 + eps) + eps
            ),
            'Mean': lambda x: math.log10(abs(x.mean()) + eps),
            'Maximum': lambda x: math.log10(abs(x.max()) + eps)
        }
        if feature in Qd_features:
            result = Qd_features[feature](diff_qdlin)
            return result

        # Discharge capacity fade curve features
        Qd = [max(x.discharge_capacity_in_Ah) for x in cell_data.cycle_data]
        #cycle 2 ~ cycle 10 까지
        Qd = Qd[self.low_cycle_index: self.high_cycle_index+1]

        if feature == 'ratio between max discharge capacity and early discharge capacity':  # noqa
            return Qd[0]/max(Qd)
        if feature == 'Slope of linear fit to the capacity curve':
            model = LinearRegression()
            x, y = np.arange(len(Qd))[:, None], np.array(Qd)
            model.fit(x, y)
            return model.coef_[0]
        if feature == 'Intercept of linear fit to the capacity curve':
            model = LinearRegression()
            x, y = np.arange(len(Qd))[:, None], np.array(Qd)
            model.fit(x, y)
            return model.intercept_

        # Other features
        if feature == 'Average early charge time':
            charge_time = []
            for cycle in range(4):
                cycle_data = cell_data.cycle_data[cycle]
                if cycle_data.time_in_s is not None:
                    charge_time.append(get_charge_time(
                        np.array(cycle_data.current_in_A),
                        np.array(cycle_data.time_in_s)
                    ))
            result = np.mean(charge_time) if len(charge_time) else 0.
            return np.log(result + eps)
        

    def open_dataset(self, feature_transform=True):
        train_features=pickle.load(open(os.path.join(train_dict, f'{self.test_name}_train_features.pkl'),'rb'))
        train_labels=pickle.load(open(os.path.join(train_dict, f'{self.test_name}_train_labels.pkl'),'rb'))
        test_features=pickle.load(open(os.path.join(test_dict, f'{self.test_name}_test_features.pkl'),'rb'))
        test_labels=pickle.load(open(os.path.join(test_dict, f'{self.test_name}_test_labels.pkl'),'rb'))

        if feature_transform==False:
            self.feature_transformation=None
            
        dataset = DataBundle(
            train_features, train_labels, test_features, test_labels,
            feature_transformation=self.feature_transformation,
            label_transformation=self.label_transformation
        )
        return dataset

#!!라벨링 되는 사이클은 실제 사이클보다 1 높음!!
    def label_annotation(self, cell_data: BatteryData) -> torch.Tensor:
        #short_circuit_cycle은 index가 아닌 cycle number이고 min cycle number로부터 데이터를 사용하므로
        #데이터가 하나 더 추가되야함. 하지만 short circuit cycle은 데이터에서 제외하므로 -1을 해야함
        data_len=cell_data.short_circuit_cycle-(self.min_cycle_index+1)+1-1
        label=[0]*(data_len-self.cyc_interval)
        label.extend([1]*self.cyc_interval)
        return torch.tensor(label)
