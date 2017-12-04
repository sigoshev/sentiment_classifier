from sentiment_classifier import SentimentClassifier

clf = SentimentClassifier()

prediction = clf.get_prediction_message('3/4 way through the first disk we played on it (naturally on 31 days after \
                                        purchase) the dvd player froze')

print(prediction[0])
