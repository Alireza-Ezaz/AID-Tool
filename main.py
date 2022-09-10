import pandas as pd
import numpy as np
import time


def ts_to_time(ts):
    # Convert timestamp to time.
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
    # Formally, given all the spans in the cloud system over a long
    # period T, we first initiate S × N empty bins of the predefined
    # size τ = 1 minute . S is the number of microservices. N, determined by T / τ,
    # is the number of bins. Then we distribute all spans into
    # different bins according to their timestamp and service name .
    # After that, we calculate the KPIS for each bin.
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
    return tmpdf, unique_services


# Section 1-1: Candidate selection
# Given the raw traces, we first
# generate a set of candidate service pairs (P, C) where service
# P directly invokes service C and P and C are different services.
print('Loading dataset...')
dataset = load_dataset('status_1min_20210411.csv.xz')
print('Dataset loaded.\n')

print('Creating candidate pairs...')
dependency_candidates = create_candidate_pairs(dataset)
child_set = set(map(lambda x: x['c'], dependency_candidates))
parent_set = set(map(lambda x: x['p'], dependency_candidates))
print('Number of dependency candidates: {}'.format(len(dependency_candidates)))
print('Number of child services: {}'.format(len(child_set)))
print('Number of parent services: {}'.format(len(parent_set)))
print('Number of services in both child and parent sets: {}'.format(len(child_set & parent_set)))
print('Number of services in child set but not in parent set: {}'.format(len(child_set - parent_set)))
print('Number of services in parent set but not in child set: {}\n'.format(len(parent_set - child_set)))
# with open('dependency-candidates.csv', 'w') as f:
#     f.write('parent_id,child_id,cnt\n')
#     for item in dependency_candidates:
#         f.write("%s,%s,%s\n" % (item['p'], item['c'], item['cnt']))

# Section 1-2: Candidate Filtering
# Only keep candidate calls whose parents appear as others' children and sort them by the number of calls.
filtered_candidates = sorted(filter(lambda x: x['p'] in child_set, dependency_candidates), key=lambda x: x['cnt'],
                             reverse=True)

# Section 2-1: Status generation
# The status of one service is composed
# of three aspects of dependency, i.e., number of invocations,
# duration of invocations, error of invocations. Each aspect of
# the service’s status contains one or more Key Performance
# Indicators (KPIs),
print('Generating service status...')
modified_dataset, unique_services = create_status(dataset)
print('Number of unique services: {}\n'.format(len(unique_services)))
# with open('service-status.csv', 'w') as f:
#     modified_dataset.to_csv(f)

# Section 2-2: Extracting KPIs(Key Performance Indicators)
#  If a service is not invoked in a particular bin (i.e.,
# the corresponding bin is empty), all the KPIs will be zero. In
# the end, we get the KPIs of every service M at every period
# t. Ordering the bins by t, we get three time series of KPIs for each cloud service
KPIs = list(modified_dataset.columns)
# with open('KPIs.csv', 'w') as f:
#     f.write('KPIs\n')
#     for item in KPIs:
#         f.write("%s\n" % item)
