from sklearn.externals import joblib
import eli5
from eli5.formatters.html import format_as_html


class SentimentClassifier:
    def __init__(self):
        self.pipeline = joblib.load('./models/TfIdfVect_stopwords_LogRegCV_model.pkl')
        self.classes_dict = {0: 'negative', 1: 'positive', -1: 'prediction_error'}

    def predict(self, text):
        try:
            class_prediction = self.pipeline.predict([text])[0]
            eli5_prediction = eli5.explain_prediction(self.pipeline.steps[1][1], text,
                                                      vec=self.pipeline.steps[0][1], top=10,
                                                      target_names=['negative', 'positive'])
            return class_prediction, eli5_prediction
        except:
            print('prediction error')
            return -1, 0.8

    def get_prediction_message(self, text):
        class_prediction, eli5_prediction = self.predict(text)
        prediction_message = self.classes_dict[class_prediction]
        return prediction_message, format_as_html(eli5_prediction)
