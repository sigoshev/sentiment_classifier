from sentiment_classifier import SentimentClassifier
import time
from flask import Flask, render_template, request
from flask import redirect, url_for
from flask_bootstrap import Bootstrap

app = Flask(__name__)
Bootstrap(app)

print("Preparing classifier")
start_time = time.time()
classifier = SentimentClassifier()
print('Classifier is ready')
print(time.time() - start_time, 'seconds')


@app.route("/")
def index():
    return redirect(url_for("sentiment"))


@app.route("/sentiment", methods=["POST", "GET"])
def sentiment(text="", prediction_message="", eli5_prediction=""):
    if request.method == "POST":
        text = request.form["text"]
        print(text)
        prediction_message, eli5_prediction = classifier.get_prediction_message(text)
        print(prediction_message)

    eli5_prediction = '<div class="col-xs-8">' + eli5_prediction + "</div>"
    return render_template('bootstrap.html', text=text,
                           prediction_message=prediction_message) + '\n' + eli5_prediction


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80, debug=False)
