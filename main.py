import pandas as pd
import numpy as np
import time


# Convert timestamp to time.
def ts_to_time(ts):
    time_array = time.localtime(ts)
    result = time.strftime("%Y-%m-%d %H:%M", time_array)
    return result


def load_dataset(dataset_name):
    # Reading a comma-separated values (csv) file into DataFrame.
    trace = pd.read_csv(dataset_name)

    # Fill NaN values with 0.
    trace['parent_csvc_name'].fillna("Source")

    # Fill NaN values with 0.
    trace['parent_cmpt_name'].fillna("Source")

    # ID is the combination of parent_csvc_name and parent_cmpt_name.
    trace['parent_id'] = trace['parent_csvc_name'] + "::" + trace['parent_cmpt_name']
    trace['child_id'] = trace['child_csvc_name'] + "::" + trace['child_cmpt_name']

    # Drop rows with same parent_id and child_id(self invocations).
    trace = trace[trace['parent_id'] != trace['child_id']]

    # Drop columns with parent_csvc_name, parent_cmpt_name, child_csvc_name, child_cmpt_name.
    trace.drop(columns=['parent_csvc_name', 'parent_cmpt_name', 'child_csvc_name', 'child_cmpt_name'], inplace=True)
    return trace


def create_candidate_pairs(trace):
    # print(trace.groupby(['parent_id', 'child_id']).agg({'call_num_sum': np.sum}))
    trace = trace.groupby(['parent_id', 'child_id']).agg({'call_num_sum': np.sum}).reset_index()
    candidate_pairs = []

    for i in range(trace.shape[0]):  # trace.shape[0] is the number of rows in the DataFrame.
        candidate_pairs.append({
            'c': trace.iloc[i]['child_id'],
            'p': trace.iloc[i]['parent_id'],
            'cnt': trace.iloc[i]['call_num_sum']
        })
    return candidate_pairs


def create_status(ds):
    unique_services = list(ds['child_id'].unique())  # get unique child IDs

    ds['from_duration_sum'] = ds['from_duration_avg'] * ds['call_num_sum']
    ds['to_duration_sum'] = ds['to_duration_avg'] * ds['call_num_sum']
    ds['from_err_num_sum'] = ds['from_err_num_avg'] * ds['call_num_sum']
    ds['to_err_num_sum'] = ds['to_err_num_avg'] * ds['call_num_sum']
    # convert timestamp to time
    ds['ts'] = ds['ts'].apply(ts_to_time)

    tmpdf = ds.groupby(['child_id', 'ts']).agg({
        'call_num_sum': np.sum,
        'from_duration_sum': np.sum,
        'from_duration_max': np.max,
        'to_duration_sum': np.sum,
        'to_duration_max': np.max,
        'from_err_num_sum': np.sum,
        'from_err_num_max': np.max,
        'to_err_num_sum': np.sum,
        'to_err_num_max': np.max
    })
    tmpdf['from_duration_avg'] = tmpdf['from_duration_sum'] / \
                                 tmpdf['call_num_sum']
    tmpdf['to_duration_avg'] = tmpdf['to_duration_sum'] / \
                               tmpdf['call_num_sum']
    tmpdf['from_err_rate'] = tmpdf['from_err_num_sum'] / \
                             tmpdf['call_num_sum']
    tmpdf['to_err_rate'] = tmpdf['to_err_num_sum'] / \
                           tmpdf['call_num_sum']
    tmpdf.drop(columns=['from_duration_sum',
                        'to_duration_sum',
                        'from_err_num_sum',
                        'to_err_num_sum'], inplace=True)
    # print(tmpdf.reset_index())
    # with open('service-status.csv', 'w') as f:
    #     tmpdf.to_csv(f)
    return tmpdf, unique_services


# Section 1: Candidate selection
# Given the raw traces, we first
# generate a set of candidate service pairs (P, C) where service
# P directly invokes service C and P and C are different services.
dataset = load_dataset('status_1min_20210411.csv.xz')
dependency_candidates = create_candidate_pairs(dataset)

# Section 2-1: Status generation
# The status of one service is composed
# of three aspects of dependency, i.e., number of invocations,
# duration of invocations, error of invocations. Each aspect of
# the service’s status contains one or more Key Performance
# Indicators (KPIs),
modified_dataset, unique_services = create_status(dataset)

# Section 2-2: Extracting KPIs(Key Performance Indicators)
KPIs = list(modified_dataset.columns)

print(*KPIs, sep='\n')
