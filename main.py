import pandas as pd
import numpy as np


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
    print(trace.groupby(['parent_id', 'child_id'])
          .agg({'call_num_sum': np.sum}))
    trace = trace.groupby(['parent_id', 'child_id']).agg({'call_num_sum': np.sum}).reset_index()
    # print(trace)
    candidate_pairs = []

    for i in range(trace.shape[0]):  # trace.shape[0] is the number of rows in the DataFrame.
        candidate_pairs.append({
            'c': trace.iloc[i]['child_id'],
            'p': trace.iloc[i]['parent_id'],
            'cnt': trace.iloc[i]['call_num_sum']
        })
    return candidate_pairs


candidate_pairs = create_candidate_pairs(load_dataset('status_1min_20210411.csv.xz'))
