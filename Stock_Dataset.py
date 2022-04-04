# import sys
# sys.path.append('C:\\Users\\USER\\PycharmProjects\\AdvLSTM')
import torch

class StockDataset:
    def __init__(self, data):
        self.data_ = data

    def __len__(self):
        return len(self.data_)

    def __getitem__(self, idx):
        data = self.data_[idx]
        X = data[0]
        y = data[1]

        return X, y  ## 데이터별 리스트
