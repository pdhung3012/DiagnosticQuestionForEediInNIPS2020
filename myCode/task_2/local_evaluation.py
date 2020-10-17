#!/usr/bin/env python
import sys
import os
import os.path
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import argparse
from pdb import set_trace

# as per the metadata file, input and output directories are the arguments
parser = argparse.ArgumentParser(description='Parse input and output paths')
parser.add_argument('--input_dir', default='../data/test_input', type=str)
parser.add_argument('--submission_dir', default='../data/test_output', type=str)
parser.add_argument('--output_dir', default='../data/test_output', type=str)
parser.add_argument('--ref_data', default='valid_task_1_2.csv', type=str)
parser.add_argument('--submit_data', default='test_submission_task_2.csv', type=str)
args = parser.parse_args()
input_dir = args.input_dir
submission_dir = args.submission_dir
output_dir = args.output_dir
submission_file_name = args.submit_data
reference_file_name = args.ref_data

# make the output dir if not exist
if not os.path.isdir(output_dir):
    os.mkdir(output_dir) 

# unzipped submission data is always in the 'res' subdirectory
# https://github.com/codalab/codalab-competitions/wiki/User_Building-a-Scoring-Program-for-a-Competition#directory-structure-for-submissions
# TODO: this file name needs to change
submission_path = os.path.join(submission_dir, submission_file_name)
if not os.path.exists(submission_path):
    message = "Expected submission file '{0}', found files {1}"
    sys.exit(message.format(submission_file_name, os.listdir(submission_dir)))
submission = pd.read_csv(submission_path)

# unzipped reference data is always in the 'ref' subdirectory
# https://github.com/codalab/codalab-competitions/wiki/User_Building-a-Scoring-Program-for-a-Competition#directory-structure-for-submissions
truth = pd.read_csv(os.path.join(input_dir, reference_file_name))
# truth = pd.read_csv('/mnt/tmp/test_public.csv')

# check if the user id and question id in submission and truth match
try:
    # u_match = np.array_equal(np.array(submission.UserId), np.array(truth.UserId))
    # q_match = np.array_equal(np.array(submission.QuestionId), np.array(truth.QuestionId))
    u_match = set(truth.UserId) <= set(submission.UserId) 
    q_match = set(truth.QuestionId) <= set(submission.QuestionId) 
    if not u_match:
        message = "user id sequence does not match that the validation file"
        sys.exit(message)
    if not q_match:
        message = "user id sequence does not match that the validation file"
        sys.exit(message)
except:
    message = "problem reading the submission file; does not contain either 'UserId' or 'QuestionId' column"
    sys.exit(message)

# get values from submission that are only in the evaluation set
true_ids = truth[['UserId', 'QuestionId']]
submission = pd.merge(true_ids, submission, on=['UserId','QuestionId'])[['UserId','QuestionId','AnswerValue']]

# sort the truth and get only those indices in the evaluation set
truth = truth.sort_values(by =['UserId', 'QuestionId'])
submission = submission.sort_values(by =['UserId', 'QuestionId'])
if list(submission.UserId) != list(truth.UserId) and list(submission.QuestionId) != list(truth.QuestionId):
    message = 'user id and question id does not match those in evaluation set'
    sys.exit(message)

# compute score
score = np.sum(np.array(submission.AnswerValue)==np.array(truth.AnswerValue)) / len(submission.AnswerValue)
conf_mtx = confusion_matrix(np.array(truth.AnswerValue), np.array(submission.AnswerValue))#, normalize=True) #/ len(submission.AnswerValue)
conf_mtx = conf_mtx / conf_mtx.sum(axis=1)
pred = ['pred_1', 'pred_2', 'pred_3', 'pred_4']
true = ['true_1', 'true_2', 'true_3', 'true_4']
df = pd.DataFrame(conf_mtx, index=true, columns=pred)
conf_mtx_str = df.to_html()

# the scores for the leaderboard must be in a file named "scores.txt"
# https://github.com/codalab/codalab-competitions/wiki/User_Building-a-Scoring-Program-for-a-Competition#directory-structure-for-submissions
with open(os.path.join(output_dir, 'scores_task_2.txt'), 'w') as output_file:
    output_file.write("score:{0}\n".format(score))

# output detailed results
with open(os.path.join(output_dir, 'scores_task_2.html'), 'w') as output_file:
    htmlString = '''<!DOCTYPE html>
                    <html>
                    <p>phase 2: classification accuracy, predict students' actual answer choice</p>
                    </br>
                    <p>overall accuracy: {}</p>
                    </br>
                    <p>confusion matrix (normalized): </p>
                    {}
                    </html>'''.format(score, conf_mtx_str)
    output_file.write(htmlString)