from sklearn.externals import joblib
import eli5
from eli5.formatters.html import format_as_html


class SentimentClassifier:
    def __init__(self):
        self.pipeline = joblib.load('./TfIdfVect_LogRegCV_model.pkl')
        self.classes_dict = {0: 'negative', 1: 'positive', -1: 'prediction_error'}

    def predict_text(self, text):
        try:
            return self.pipeline.predict([text])[0], \
                   eli5.explain_prediction(self.pipeline.steps[1][1], text, vec=self.pipeline.steps[0][1], top=10,
                                           target_names=['negative', 'positive'])
        except:
            print('prediction error')
            return -1, 0.8

    def get_prediction_message(self, text):
        prediction = self.predict_text(text)
        class_prediction = prediction[0]
        eli5_prediction = prediction[1]
        prediction_message = self.classes_dict[class_prediction]
        return prediction_message, format_as_html(eli5_prediction)
