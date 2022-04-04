from re import L
import pandas as pd
import numpy as np
import torch.nn as nn
import torch
import os
from sklearn import preprocessing
import copy

class stock_csv_read:
    def __init__(self, x_frames, y_frames):
        self.x_frames = x_frames
        self.y_frames = y_frames
        self.data_list = os.listdir("C:/Users/USER/JupyterProjects/bilstm_attention_ti_cor/data/kdd17/price_long_50_/")
        self.stock_data = self.data_loader()

        
    def data_loader(self):
        stock_list = []
        for data in self.data_list:
            stock_data = pd.read_csv("C:/Users/USER/JupyterProjects/bilstm_attention_ti_cor/data/kdd17/price_long_50_/"+ data ,header=0,index_col="Date")

            open = stock_data.loc[:,"Open"]
            high = stock_data.loc[:,"High"]
            low = stock_data.loc[:,"Low"]
            close = stock_data.loc[:,"Adj Close"]
            volume = stock_data.loc[:,"Volume"]

            upndown = (stock_data.loc[:, "Adj Close"] - stock_data.loc[:, "Adj Close"].shift(periods=-1, axis=0))
            change = (((stock_data.loc[:, "Adj Close"] - stock_data.loc[:, "Adj Close"].shift(periods=-1, axis=0))/stock_data.loc[:, "Adj Close"].shift(periods=-1, axis=0))*100)
            tgt = np.where(stock_data.loc[:, "Adj Close"] >= stock_data.loc[:, "Adj Close"].shift(periods=-1, axis=0), 1.0, 0.0)
            
            # technical_indicator

            # 1) 10일 이동평균
            ten_day_ma = copy.copy(stock_data.loc[:, "Adj Close"])
            for i in range(9):
                ten_day_ma += stock_data.loc[:, "Adj Close"].shift(periods = -i-1, axis = 0)
            
            ten_day_ma = ten_day_ma/10 ## 마지막 nan 9개가 생김
                
            # 2) 10일 가중 이동평균
            w_ten_day_ma = copy.copy(stock_data.loc[:, "Adj Close"])
            w_ten_day_ma_10 = copy.copy(w_ten_day_ma*10)
            for i in range(9):
                w_ten_day_ma_10 += (9-i)*stock_data.loc[:, "Adj Close"].shift(periods = -i-1, axis = 0)
            
            wma = w_ten_day_ma_10/((10*9)/2)

            del w_ten_day_ma
            del w_ten_day_ma_10

            # 3) momentum
            momentum = stock_data.loc[:, "Adj Close"] - stock_data.loc[:, "Adj Close"].shift(periods = -10, axis = 0)


            # 4) stochastic_K%
            init_low = copy.copy(stock_data.loc[:, "Low"])
            init_high = copy.copy(stock_data.loc[:, "High"])
            
            for i in range(9):
                second_low = copy.copy(stock_data.loc[:, "Low"].shift(periods = -i-1, axis = 0))
                second_high = copy.copy(stock_data.loc[:, "High"].shift(periods = -i-1, axis = 0))

                if i == 0:
                    lows = pd.concat([init_low,second_low],axis = 1,ignore_index=True)
                    highs = pd.concat([init_high,second_high],axis = 1,ignore_index=True)
                else:
                    lows = pd.concat([lows,second_low],axis = 1,ignore_index=True)
                    highs = pd.concat([highs,second_high],axis = 1,ignore_index=True)
            
            row_low = lows.min(axis=1)
            row_high = highs.max(axis=1)

            stochastic_K = ((stock_data.loc[:,"Close"]-row_low)/(row_high-row_low))*100

            del row_low
            del row_high
            del lows
            del highs
            del second_low
            del second_high
            del init_low
            del init_high

            # 4) stochastic_D%
            stochastic_D = copy.copy(stochastic_K)
            for i in range(9):
                stochastic_D += stochastic_K.shift(periods = -i-1, axis = 0)
            stochastic_D = stochastic_D/10    

            # RSI
            difference = stock_data.loc[:, "Adj Close"] - stock_data.loc[:, "Adj Close"].shift(periods = -1, axis = 0)

            u = abs(difference.where(difference>0,0))
            d = abs(difference.where(difference<0,0))
            init_u = copy.copy(u)
            init_d = copy.copy(d)

            for i in range(9):
                init_u += u.shift(periods = -i-1, axis = 0)
                init_d += d.shift(periods = -i-1, axis = 0)
            
            AU = init_u/10
            AD = init_d/10

            RSI = 100-100/(1+AU/AD)

            del AU
            del AD
            del init_u
            del init_d
            del u
            del d
            del difference

            df1 = stock_data.loc[:, "Adj Close"]

            df1 = df1.iloc[::-1]
            

            ema_12 = df1.ewm(span=12,min_periods=11,adjust = True).mean()
            ema_26 = df1.ewm(span=26,min_periods=25,adjust = True).mean()

            # MACD
            MACD =  ema_12 - ema_26

            MACD = MACD[::-1]

            del ema_12
            del ema_26
            del df1

            # Larry_williams_R 
            LWR = ((stock_data.loc[:,"High"]-stock_data.loc[:,"Close"])/(stock_data.loc[:,"High"]-stock_data.loc[:,"Low"]))*100

            # A_D_Oscillator 
            A_D = (stock_data.loc[:,"High"]-stock_data.loc[:,"Close"].shift(periods=-1, axis=0))/(stock_data.loc[:,"High"]-stock_data.loc[:,"Low"])
            
            # CCI
            MT = stock_data.loc[:,"High"]+stock_data.loc[:,"Low"]+stock_data.loc[:,"Close"]/3
            SMT = copy.copy(MT)
            for i in range(9):
                SMT += MT.shift(periods=-1-i, axis=0)
            
            SMT = SMT/10

            DT = abs(MT.shift(periods=-9, axis=0) - SMT)
            for i in range(9):
                DT += abs(MT.shift(periods=-i, axis=0) - SMT)
            
            DT = DT/10

            CCI = (MT -SMT)/(0.015*DT)
            
            del MT
            del DT
            del SMT

            d_len = len(RSI)

            df = np.column_stack((open, high, low, close, volume,upndown,change, ten_day_ma,wma, momentum, stochastic_K, stochastic_D, RSI, MACD, LWR, A_D, CCI))
            
            # scaler  = preprocessing.StandardScaler().fit(df)
            # scaled_df = scaler.transform(df)
            # data = np.column_stack((scaled_df,tgt))[:(d_len-24)]
            data = pd.DataFrame(np.column_stack((df,tgt))[:(d_len-24)][::-1].copy())
            stock_list.append(data)
            
        return stock_list   ## 50 종목의 주식이 들어있는 list
    
    def spliter(self, data):
        self.dd = data
        data_list = []
        for i in range(len(self.dd)-self.x_frames-self.y_frames+1):
            xy = []
            
            X = self.dd.iloc[i : i+self.x_frames, 0:17]
            scaler  = preprocessing.StandardScaler().fit(X)
            X = scaler.transform(X)
            y = self.dd.iloc[i+self.x_frames : i+self.x_frames+self.y_frames, 17:].to_numpy()
            

            xy.append(X)
            xy.append(y)
            data_list.append(xy)
        return data_list
    
    def cv_split(self):
        stocks_data = self.stock_data ## 50개 주식이 들어있는 list
        
        stock_50_data = []
        train_50_data = []
        validation_50_data = []
        test_50_data = []
        for stock_data in stocks_data:
            data_len = len(stock_data)
            mok = data_len//19
            
            adder = 0
            train_list = []
            validation_list = []
            test_list = []
            for i in range(10):
                sp_data = stock_data.iloc[0+adder:10*mok+adder,:]
                

                train_sp_data = sp_data[0:8*mok]
                validation_sp_data = sp_data[8*mok:9*mok]
                test_sp_data = sp_data[9*mok:10*mok]
                
                train_sp_data_ = self.spliter(train_sp_data)
                validation_sp_data_ = self.spliter(validation_sp_data)
                test_sp_data_ = self.spliter(test_sp_data)

                adder += mok
                train_list.append(train_sp_data_)
                validation_list.append(validation_sp_data_)
                test_list.append(test_sp_data_)
                
            train_50_data.append(train_list)
            validation_50_data.append(validation_list)
            test_50_data.append(test_list)
        
        stock_50_data.append(train_50_data)
        stock_50_data.append(validation_50_data)
        stock_50_data.append(test_50_data)

        return stock_50_data, self.data_list

