from pathlib import Path
if __name__== '__main__':
    from directory import processed_dict_1, processed_dict_2
else:
    from tools.directory import processed_dict_1, processed_dict_2
import os
import random
from tools.setting import seed
from tqdm import tqdm
from tools.tool_battery_data import BatteryData


def train_test_split(train_list, test_list):
    pbar = tqdm(train_list, desc='Reading train data')
    train_cells = [BatteryData.load(path) for path in pbar]
    pbar = tqdm(test_list, desc='Reading test data')
    test_cells= [BatteryData.load(path) for path in pbar] 
    return train_cells, test_cells

def split(SEED=seed):
    random.seed(SEED)
    file_list_1=os.listdir(processed_dict_1)
    file_list_1=[full_name.replace('.pkl','') for full_name in file_list_1]
    test_cell_1=random.sample(file_list_1, 2)
  
    file_list_2=os.listdir(processed_dict_2)
    file_list_2=[full_name.replace('.pkl','') for full_name in file_list_2]
    test_cell_2=random.sample(file_list_2, 2)
    
    file_list_1+=file_list_2
    test_cell_1+=test_cell_2
    test_ids=test_cell_1
    train_ids=[cell for cell in file_list_1 if cell not in test_ids]

    cell_data_path=[processed_dict_1, processed_dict_2]

    _file_list = []
    for path in cell_data_path:
        path = Path(path)
        assert path.exists(), path

        if path.is_dir():
            _file_list += list(path.glob('*.pkl'))

    # NOTE: the filename should be the cell IDs

    # Build a map from train_id to file_path
    path_map = {}
    for cell_path in _file_list:
        cell_path = Path(cell_path)
        cell_id = cell_path.stem
        path_map[cell_id] = cell_path

    train_list = [path_map[cell] for cell in train_ids]
    test_list = [path_map[cell] for cell in test_ids]

    assert len(train_list) == len(train_ids)
    assert len(test_list) == len(test_ids)

    train_cells, test_cells=train_test_split(train_list, test_list)

    return train_cells, test_cells

if __name__== '__main__':
   train_cells, test_cells=split()
# print(train_list)
# print(test_list)
