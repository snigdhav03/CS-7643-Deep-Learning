import os.path

import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score, \
    precision_score, recall_score

from src.const import project_dir

import matplotlib.pyplot as plt
import numpy as np


class ClassificationStatistics:

    def __init__(self, model, model_name, y_predicted, y_predicted_indices, y_actual,
                 prediction_df: pd.DataFrame, averaging='weighted'):
        self._model = model
        self._model_name = model_name
        self._y_predicted_proba = y_predicted
        self._y_predicted = y_predicted_indices
        self._y_actual = y_actual
        self._averaging = averaging
        self.prediction_df = prediction_df
        self._model_name = (' '.join(word.capitalize() for word in model_name.split('_')))

    def get_confusion_matrix(self):
        cm = confusion_matrix(self._y_actual, self._y_predicted)
        plt.imshow(cm, interpolation='nearest', cmap='GnBu')
        plt.title('Confusion Matrix for ' + self._model_name)
        plt.colorbar()
        plt.xticks([0, 1], ['Away Team', 'Home Team'])
        plt.yticks([0, 1], ['Away Team', 'Home Team'])
        plt.xlabel('Predicted Winner')
        plt.ylabel('Actual Winner')
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, cm[i, j], horizontalalignment="center",
                         color="white" if cm[i, j] > cm.max() / 2 else "black")
        plt.tight_layout()
        if not os.path.exists(project_dir + '/results/' + self._model_name):
            os.mkdir(project_dir + '/results/' + self._model_name)
        plt.savefig(project_dir + '/results/' + self._model_name + '/confusion_matrix.png')
        plt.close()
        return cm

    def get_classification_report(self):
        return classification_report(self._y_actual, self._y_predicted)

    def get_accuracy_score(self):
        return accuracy_score(self._y_actual, self._y_predicted)

    def get_f1_score(self):
        return f1_score(self._y_actual, self._y_predicted, average=self._averaging)

    def get_precision_score(self):
        return precision_score(self._y_actual, self._y_predicted, average=self._averaging)

    def get_recall_score(self):
        return recall_score(self._y_actual, self._y_predicted, average=self._averaging)

    def save_results(self):
        os.makedirs(project_dir + '/results/' + self._model_name, exist_ok=True)
        with open(project_dir + '/results/' + self._model_name + '/classification_report.txt', 'w') as f:
            print('Results for Model: ', self._model_name, file=f)
            # print('Confusion Matrix: \n', self.get_confusion_matrix(), file=f)
            # print('Classification Report: \n', self.get_classification_report(), file=f)
            print('Accuracy Score: ', self.get_accuracy_score(), file=f)
            print('F1 Score: ', self.get_f1_score(), file=f)
            print('Precision Score: ', self.get_precision_score(), file=f)
            print('Recall Score: ', self.get_recall_score(), file=f)
        self.prediction_df.to_csv(project_dir + '/results/' + self._model_name + '/predictions.csv', index=False)
