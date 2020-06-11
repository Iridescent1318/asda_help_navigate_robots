import pandas as pd
import numpy as np
import csv
import time
import scipy.stats

FEAT_EXTRACT = 0

def series_fft(x):
    return (np.abs(np.fft.fft(x)) / len(x))[range(int(len(x)/2))]

def scipy_skew(x):
    return scipy.stats.skew(x)

def scipy_kurtosis(x):
    return scipy.stats.kurtosis(x)

if __name__ == "__main__":

    read_start = time.perf_counter()
    x_train_df = pd.read_csv('X_train.csv')
    read_time = time.perf_counter() - read_start
    print("Load X_train.csv total time:{:.6f} second(s).".format(read_time))

    if FEAT_EXTRACT:
        pre_proc_start = time.perf_counter()
        x_train_feats = pd.DataFrame()
        x_train_df['total_angular_velocity'] = (x_train_df['angular_velocity_X']**2 
                                            + x_train_df['angular_velocity_Y']**2
                                            + x_train_df['angular_velocity_Z']**2 )** 0.5
        x_train_df['total_linear_acceleration'] = (x_train_df['linear_acceleration_X']**2 
                                                + x_train_df['linear_acceleration_Y']**2
                                                + x_train_df['linear_acceleration_Z']**2 )** 0.5
        x_train_df['acc_div_velocity'] = x_train_df['total_angular_velocity'] / x_train_df['total_linear_acceleration']
        pre_proc_time = time.perf_counter() - pre_proc_start
        print("Pre-processing total time:{:.6f} second(s).".format(pre_proc_time))

        feats_extract_start = time.perf_counter()
        for c in x_train_df.columns:
            if c in ['row_id', 'series_id', 'measurement_number']:
                continue
            x_train_feats[c + '_mean'] = x_train_df.groupby(['series_id'])[c].mean()
            x_train_feats[c + '_std']  = x_train_df.groupby(['series_id'])[c].std()
            x_train_feats[c + '_max']  = x_train_df.groupby(['series_id'])[c].max()
            x_train_feats[c + '_min']  = x_train_df.groupby(['series_id'])[c].min()
            x_train_feats[c + '_median']  = x_train_df.groupby(['series_id'])[c].median()
            x_train_feats[c + '_skewness'] = x_train_df.groupby(['series_id'])[c].apply(scipy_skew)
            x_train_feats[c + '_kurtosis'] = x_train_df.groupby(['series_id'])[c].apply(scipy_kurtosis)
            x_train_feats[c + '_maxOverMin']  = x_train_feats[c + '_max'] / (x_train_feats[c + '_min'] + 1e-10)
            x_train_feats[c + '_meanAbsChange']  = x_train_df.groupby(['series_id'])[c].apply(lambda x: np.mean(np.abs(np.diff(x))))
            x_train_feats[c + '_meanChangeOfAbsChange']  = x_train_df.groupby(['series_id'])[c].apply(lambda x: np.mean(np.diff(np.abs(np.diff(x)))))
            x_train_feats[c + '_absMax']  = x_train_df.groupby(['series_id'])[c].apply(lambda x: np.max(np.abs(x)))
            x_train_feats[c + '_absMin']  = x_train_df.groupby(['series_id'])[c].apply(lambda x: np.min(np.abs(x)))
            x_train_feats[c + '_absMean'] = x_train_df.groupby(['series_id'])[c].apply(lambda x: np.mean(np.abs(x)))
            x_train_feats[c + '_absStd'] = x_train_df.groupby(['series_id'])[c].apply(lambda x: np.std(np.abs(x)))
            # if c in ['total_angular_velocity', 'total_linear_acceleration']:
            #     x_train_feats[c + '_fftMean'] = x_train_df.groupby(['series_id'])[c].apply(lambda x: np.std(np.abs(x)))
        feats_extract_time = time.perf_counter() - feats_extract_start

        print("Feature extracting total time:{:.6f} second(s).".format(feats_extract_time))
        print(x_train_feats.head())
        x_train_feats.to_csv('x_train_feats.csv', index=True)
    else:
        x_train_original = pd.DataFrame()
        for c in x_train_df.columns:
            if c in ['row_id', 'series_id', 'measurement_number']:
                continue
            new_column = x_train_df[c].values.tolist()
            for j in range(128):
                x_train_original[c + '_' + str(j)] = [new_column[x] for x in range(3810*128) if x%128 == j]
        print(x_train_original.head())
        x_train_original.to_csv('x_train_original.csv', index=True)
    pass