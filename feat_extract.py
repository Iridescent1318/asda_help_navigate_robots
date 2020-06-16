import pandas as pd
import numpy as np
import math
import csv
import time
import scipy.stats

FILE_NAME = 'X_train'
FEAT_EXTRACT = 1
EULER = 1
ORIENT = 1

def quat_to_euler(x, y, z, w):
    t0 = 2. * (w*x + y*z)
    t1 = 1. - 2. * (x*x + y*y)
    X = math.atan2(t0, t1)

    t2 = 2. * (w*y - x*z)
    if t2 > 1.:
        t2 = 1.0
    if t2 < -1.:
        t2 = -1.0
    Y = math.asin(t2)

    t3 = 2. * (w*z + x*y)
    t4 = 1. - 2. * (y*y + z*z)
    Z = math.atan2(t3, t4)

    return X, Y, Z

def series_fft(x):
    return (np.abs(np.fft.fft(x)) / len(x))[range(int(len(x)/2))]

def topk_val(x, k):
    if k > len(x):
        return None
    x_sort = np.sort(x)[::-1]
    return x_sort[0:k]

def topk_arg(x, k):
    if k > len(x):
        return None
    x_sort = np.argsort(x)[::-1]
    return x_sort[0:k]

def scipy_skew(x):
    return scipy.stats.skew(x)

def scipy_kurtosis(x):
    return scipy.stats.kurtosis(x)

if __name__ == "__main__":

    read_start = time.perf_counter()
    x_train_df = pd.read_csv(FILE_NAME + '.csv')
    read_time = time.perf_counter() - read_start
    print("Load X_train.csv total time:{:.6f} second(s).".format(read_time))

    if FEAT_EXTRACT:
        if EULER:
            q2e_start = time.perf_counter()
            orient_x = x_train_df['orientation_X'].values.tolist()
            orient_y = x_train_df['orientation_Y'].values.tolist()
            orient_z = x_train_df['orientation_Z'].values.tolist()
            orient_w = x_train_df['orientation_W'].values.tolist()

            e_x = []
            e_y = []
            e_z = []

            for ox, oy, oz, ow in zip(orient_x, orient_y, orient_z, orient_w):
                ex, ey, ez = quat_to_euler(ox, oy, oz, ow)
                e_x.append(ex)
                e_y.append(ey)
                e_z.append(ez)
            
            x_train_df['euler_X'] = e_x
            x_train_df['euler_Y'] = e_y
            x_train_df['euler_Z'] = e_z
            

            q2e_time = time.perf_counter() - q2e_start
            print("Quarternion-to-Euler total time:{:.6f} second(s).".format(q2e_time))

        if ORIENT == 0:
            del x_train_df['orientation_X']
            del x_train_df['orientation_Y']
            del x_train_df['orientation_Z']
            del x_train_df['orientation_W']

        pre_proc_start = time.perf_counter()
        x_train_feats = pd.DataFrame()
        x_train_df['total_angular_velocity'] = (x_train_df['angular_velocity_X']**2 
                                            + x_train_df['angular_velocity_Y']**2
                                            + x_train_df['angular_velocity_Z']**2 )** 0.5
        x_train_df['total_linear_acceleration'] = (x_train_df['linear_acceleration_X']**2 
                                                + x_train_df['linear_acceleration_Y']**2
                                                + x_train_df['linear_acceleration_Z']**2 )** 0.5
        if EULER:
            x_train_df['total_euler'] = (x_train_df['euler_X']**2 
                                       + x_train_df['euler_Y']**2
                                       + x_train_df['euler_Z']**2 )** 0.5
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
            x_train_feats[c + '_range'] = x_train_feats[c + '_max'] - x_train_feats[c + '_min']
            x_train_feats[c + '_median']  = x_train_df.groupby(['series_id'])[c].median()
            x_train_feats[c + '_skewness'] = x_train_df.groupby(['series_id'])[c].apply(scipy_skew)
            x_train_feats[c + '_kurtosis'] = x_train_df.groupby(['series_id'])[c].apply(scipy_kurtosis)
            x_train_feats[c + '_maxOverMin']  = x_train_feats[c + '_max'] / (x_train_feats[c + '_min'] + 1e-10)
            x_train_feats[c + '_meanAbsChange']  = x_train_df.groupby(['series_id'])[c].apply(lambda x: np.mean(np.abs(np.diff(x))))
            x_train_feats[c + '_meanChangeOfAbsChange']  = x_train_df.groupby(['series_id'])[c].apply(lambda x: np.mean(np.diff(np.abs(np.diff(x)))))
            x_train_feats[c + '_q25'] = x_train_df.groupby(['series_id'])[c].quantile(0.25)
            x_train_feats[c + '_q75'] = x_train_df.groupby(['series_id'])[c].quantile(0.75)
            x_train_feats[c + '_absMax']  = x_train_df.groupby(['series_id'])[c].apply(lambda x: np.max(np.abs(x)))
            x_train_feats[c + '_absMin']  = x_train_df.groupby(['series_id'])[c].apply(lambda x: np.min(np.abs(x)))
            x_train_feats[c + '_absRange'] = x_train_feats[c + '_absMax'] - x_train_feats[c + '_absMin']
            x_train_feats[c + '_absMaxOverMin'] = x_train_feats[c + '_absMax'] / (x_train_feats[c + '_absMin'] + 1e-10)
            x_train_feats[c + '_absMean'] = x_train_df.groupby(['series_id'])[c].apply(lambda x: np.mean(np.abs(x)))
            x_train_feats[c + '_absStd'] = x_train_df.groupby(['series_id'])[c].apply(lambda x: np.std(np.abs(x)))
            # DEPRECATED
            # if c in ['total_angular_velocity', 'total_linear_acceleration', 'angular_velocity_X', 'angular_velocity_Y', 'angular_velocity_Z',
            #          'linear_acceleration_X', 'linear_acceleration_Y', 'linear_acceleration_Z']:
            #     x_fft_val_topk = x_train_df.groupby(['series_id'])[c].apply(lambda x, k=5: topk_val(series_fft(x), k)).values.tolist()
            #     x_fft_arg_topk = x_train_df.groupby(['series_id'])[c].apply(lambda x, k=5: topk_arg(series_fft(x), k)).values.tolist()
            #     for i in range(5):
            #         x_train_feats[c + '_fft_val_top_' + str(i+1)] = [c[i] for c in x_fft_val_topk]
            #         x_train_feats[c + '_fft_arg_top_' + str(i+1)] = [c[i] for c in x_fft_arg_topk]
        feats_extract_time = time.perf_counter() - feats_extract_start

        print("Feature extracting total time:{:.6f} second(s).".format(feats_extract_time))
        print(x_train_feats.head())
        x_train_feats.to_csv(FILE_NAME + '_feats.csv', index=True)
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