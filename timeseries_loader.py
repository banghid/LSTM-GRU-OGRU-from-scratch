import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


class TimeseriesLoader(object):
    
    def __init__(self, _bat_size = 12):
        # self.DIR = 'data/'
        # new dataset
        self.DIR = 'data/new data/'
        self.batch_size = _bat_size

    def load_data(self):
        # old dataset
        # bitcoin_time_series = pd.read_csv(self.DIR + "data_bitcoinity_full_daily.csv", parse_dates = ['Time'])
        # gtrend_time_series = pd.read_csv(self.DIR + "daily_gtrend_data.csv", parse_dates = ['date'])
        # added timeseries data
        # bitcoin_time_series = pd.read_csv(self.DIR + "bitcoin_price_daily.csv", parse_dates = ['Time'])
        bitcoin_time_series = pd.read_csv(self.DIR + "coin_marketcap_data_fixed.csv", parse_dates = ['date'])

        gtrend_time_series = pd.read_csv(self.DIR + "daily_gtrend_data_cmc.csv", parse_dates = ['date'])
        # block_size = pd.read_csv(self.DIR + "block_size_cut.csv", parse_dates = ['Time'])
        # mining_difficulty = pd.read_csv(self.DIR + "mining_difficulty_daily_cut.csv", parse_dates = ['Time'])
        # number_of_transaction = pd.read_csv(self.DIR + "number_of_transaction_cut.csv", parse_dates = ['Time'])
        bitcoin_time_series = bitcoin_time_series.sort_values(by='index', ascending=False)
        #joinning gtrends and bitcoin data
        dataset = bitcoin_time_series.copy()
        dataset['gtrend'] = gtrend_time_series['bitcoin']
        # dataset['block_size'] = block_size['block_size']
        # dataset['difficulty'] = mining_difficulty['difficulty']
        # dataset['number_of_transaction'] = number_of_transaction['not']

        # deleting unnecessary variable
        del gtrend_time_series
        del bitcoin_time_series
        # del block_size
        # del mining_difficulty
        # del number_of_transaction

        #dropping timestamp
        dataset = dataset.drop(columns=['date', 'index'])

        # Splitting data into train and test data
        train_size = int(len(dataset)*0.8)
        train_data = dataset.iloc[:train_size]
        test_data = dataset.iloc[train_size:]

        #scaling dataset
        scaler = MinMaxScaler().fit(dataset)
        train_scaled = scaler.transform(train_data)
        test_scaled = scaler.transform(test_data)

        print(train_scaled)

        trainX, trainY = self.sliding_window(train_scaled)
        testX, testY = self.sliding_window(test_scaled)

        return trainX, trainY, testX, testY

    def sliding_window(self, dataset ,n_future = 1, n_past = 14):
        Xs = []
        Ys = []
        for i in range(n_past, len(dataset) - n_future+1):
            Xs.append(dataset[i - n_past:i, 0:dataset.shape[1]])
            Ys.append(dataset[i + n_future - 1:i + n_future,0])

        return np.array(Xs), np.array(Ys)
    
    def create_batches(self, x, y, batch_size=100):
        Xs = []
        Ys = []
        n = len(x)/batch_size
        # print("n value is:", n)
        
        for i in range(int(n)):
            Xs.append(x[i*batch_size:(i+1)*batch_size, 0:x.shape[1]])
            Ys.append(y[i*batch_size:(i+1)*batch_size, 0:y.shape[1]])

        return np.array(Xs), np.array(Ys)

    # def create_batches(self, x, y, batch_size=100):
    #     perm = np.random.permutation(len(x))
    #     print('perm value:', perm)
    #     x = x[perm]
    #     y = y[perm]

    #     batch_x = np.array_split(x, len(x) / batch_size)
    #     batch_y = np.array_split(y, len(y) / batch_size)

    #     return batch_x, batch_y

