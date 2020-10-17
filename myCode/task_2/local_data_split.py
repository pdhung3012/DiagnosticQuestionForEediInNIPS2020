'''
this script splits the training set into two subsets for local evaluation

input: training split of the competition data
output: a 80-20 split of the input data. the ratio can be manually changed in the script
'''

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
np.random.seed(123)

data = pd.read_csv(os.path.normpath('../data/train_data/train_task_1_2.csv'))
train, valid = train_test_split(data, test_size=0.2)
test_submission_ids = valid[['UserId', 'QuestionId']]

if not os.path.isdir(os.path.normpath('../data/test_input')):
    os.mkdir(os.path.normpath('../data/test_input'))

train.to_csv(os.path.normpath('../data/test_input/train_task_1_2.csv'), index=False)
valid.to_csv(os.path.normpath('../data/test_input/valid_task_1_2.csv'), index=False)
test_submission_ids.to_csv(os.path.normpath('../data/test_input/test_submission_task_1_2.csv'), index=False)
