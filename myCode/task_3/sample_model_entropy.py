# Approximate the entropy of each question by treating each question as a categorical distribution whose class 
# probabilities we estimate from the data. Rank questions with larger approximated entropies as higher quality.

import pandas as pd
import scipy as sp
from scipy.stats import multinomial
import os
import pandas as pd
import numpy as np
import argparse

# as per the metadata file, input and output directories are the arguments
parser = argparse.ArgumentParser(description='Parse input and output paths')
parser.add_argument('--input_dir', default=os.path.normpath('../data/train_data'), type=str)
parser.add_argument('--train_data', default='train_task_3_4.csv', type=str)
parser.add_argument('--submission_template_dir', default=os.path.normpath('../submission_templates'))
parser.add_argument('--submission_template_file', default='submission_task_3.csv', type=str)
parser.add_argument('--output_dir', default=os.path.normpath('../submissions'), type=str)
parser.add_argument('--output_file', default=os.path.normpath('submission_task_3.csv'), type=str)

args = parser.parse_args()
input_dir = args.input_dir
train_data = args.train_data
submission_template_dir = args.submission_template_dir
submission_template_file = args.submission_template_file
output_dir = args.output_dir
output_file = args.output_file

if not os.path.isdir(os.path.normpath(output_dir)):
    os.makedirs(os.path.normpath(output_dir))

data_path = os.path.join(input_dir, train_data)
df = pd.read_csv(data_path)

quality = df.groupby('QuestionId')['AnswerValue'].agg(lambda x:multinomial.entropy(1,x.value_counts(normalize=True)))
ranking = quality.rank(method='first', ascending=False).astype('int16')

submission_file_path = os.path.join(submission_template_dir, submission_template_file)
submission_df = pd.read_csv(submission_file_path)
submission_df['ranking'] = ranking

output_path = os.path.join(output_dir, output_file)
submission_df.to_csv(output_path, index=False)