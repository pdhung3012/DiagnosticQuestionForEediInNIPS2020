# For each question, predict every student's corresponding answer based on whichever answer is most common for that 
# question in the training data.

import pandas as pd
import argparse
import os

# as per the metadata file, input and output directories are the arguments
parser = argparse.ArgumentParser(description='Parse input and output paths')

parser.add_argument('--train_data', default='train_task_1_2.csv', type=str)

# Default arguments for Codalab submission
parser.add_argument('--input_dir', default=os.path.normpath('../data/train_data'), type=str)
parser.add_argument('--submission_template_dir', default=os.path.normpath('../submission_templates'))
parser.add_argument('--submission_template_file', default='submission_task_1_2.csv', type=str)
parser.add_argument('--output_dir', default=os.path.normpath('../submissions'), type=str)
parser.add_argument('--output_file', default=os.path.normpath('submission_task_2.csv'), type=str)

# Default arguments for local evaluation 
# parser.add_argument('--input_dir', default=os.path.normpath('../data/test_input'), type=str)
# parser.add_argument('--submission_template_dir', default=os.path.normpath('../data/test_input'))
# parser.add_argument('--submission_template_file', default='test_submission_task_1_2.csv', type=str)
# parser.add_argument('--output_dir', default=os.path.normpath('../data/test_output'), type=str)
# parser.add_argument('--output_file', default=os.path.normpath('test_submission_task_2.csv'), type=str)

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
# Get whether the question is more commonly answered correctly or incorrectly
top_answers = df.groupby('QuestionId')['AnswerValue'].agg(lambda x:x.value_counts().index[0])
top_answers_dict = top_answers.to_dict()

test_data_path = os.path.join(submission_template_dir, submission_template_file)
test_df = pd.read_csv(test_data_path)

test_df['AnswerValue'] = test_df['QuestionId'].map(top_answers_dict)

output_path = os.path.join(output_dir, output_file)
test_df.to_csv(output_path, index=False)