from model import MyModel

class Submission:
    """
    API Wrapper class which loads a saved model upon construction, and uses this to implement an API for feature 
    selection and missing value prediction. This API will be used to perform active learning evaluation in private.
    """
    def __init__(self):
        # Load a saved model here.
        self.model = MyModel()
        self.model.load('most_popular.npy', 'num_answers.npy')

    def select_feature(self, masked_data, can_query):
        """
        Use your model to select a new feature to observe from a list of candidate features for each student in the
            input data, with the goal of selecting features with maximise performance on a held-out set of answers for
            each student.
        Args:
            masked_data (np.array): Array of shape (num_students, num_questions) containing data revealed to the model
                at the current step. Unobserved values are denoted by -1.
            can_query (np.array): Binary array of shape (num_students, num_questions), indicating which features can be 
                queried by the model in the current step.
        Returns:
            selections (list): List of ints, length num_students, containing the index of the feature selected to query 
            for each student (row) in the dataset.
        """
        # Use the loaded model to perform feature selection.
        selections = self.model.select_feature(masked_data, can_query)

        return selections

    def predict(self, masked_data):
        """
        Use your model to predict missing values in the input data.
        Args:
            masked_data (np.array): Array of shape (num_students, num_questions) containing data revealed to the model
                at the current step. Unobserved values are denoted by -1.
        Returns:
            predictions (np.array): Array of shape (num_students, num_questions) containing predictions for the
                unobserved values in `masked_data`. The values given to the observed data in this array will be ignored.
        """
        # Use the loaded model to perform missing value prediction.
        predictions = self.model.predict(masked_data)

        return predictions



