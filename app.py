from flask import Flask, render_template, request, session
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import XLNetTokenizer
from keras.preprocessing.sequence import pad_sequences
import torch



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def predict(text):
    #     labeled_df = pd.read_parquet("full_raw_data.parquet.gzip")
    #     labeled_df['sentiment'] = labeled_df['sentiment'].map({"neutral":1,"positive":2,"negative":0})
    #     train_df ,test_df = train_test_split(labeled_df,test_size=0.2)
    #     train_iter = [(label,text) for label,text in zip(train_df['sentiment'].to_list(),train_df['text'].to_list())]

    #     # Build vocabulary from tokens of training set
    tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased', do_lower_case=True)
    tokens = tokenizer.tokenize(text)
    input_id = [tokenizer.convert_tokens_to_ids(tokens)]
    MAX_LEN = 128
    input_id = pad_sequences(input_id, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
    input_id_t = torch.tensor(input_id).to(device)
    # Model class must be defined somewhere

    model = "models/pretrained1.pt"
    print("Loading:", model)

    net = torch.load(model,map_location=torch.device('cpu'))

    net = net.to(device)
    # https://pytorch.org/docs/stable/torchvision/models.html
    attention_masks = []
    for seq in input_id:
      seq_mask = [float(i > 0) for i in seq]
      attention_masks.append(seq_mask)
    mask = torch.tensor(attention_masks).to(device)




    out = net(input_id_t, token_type_ids=None, attention_mask=mask)

    classes = ["negative", "neutral", "positive"]

    prob = torch.nn.functional.softmax(out[0], dim=1)[0] * 100
    print(prob)
    _, indices = torch.sort(out[0], descending=True)
    return [(classes[idx], prob[idx].item()) for idx in indices[0][:3]]
# Sentiment analysis function using VADER
def vader_sentiment_scores(data_frame):
    # Define SentimentIntensityAnalyzer object of VADER.
    SID_obj = SentimentIntensityAnalyzer()

    # calculate polarity scores which gives a sentiment dictionary,
    # Contains pos, neg, neu, and compound scores.
    sentiment_list = []
    for row_num in range(len(data_frame)):
        sentence = data_frame['Text'][row_num]

        polarity_dict = SID_obj.polarity_scores(sentence)

        # Calculate overall sentiment by compound score
        if polarity_dict['compound'] >= 0.05:
            sentiment_list.append("Positive")

        elif polarity_dict['compound'] <= - 0.05:
            sentiment_list.append("Negative")

        else:
            sentiment_list.append("Neutral")

    data_frame['Sentiment'] = sentiment_list

    return data_frame

def transformer(data_frame):
    # Define SentimentIntensityAnalyzer object of VADER.
    #SID_obj = SentimentIntensityAnalyzer()

    # calculate polarity scores which gives a sentiment dictionary,
    # Contains pos, neg, neu, and compound scores.
    sentiment_list = []
    for row_num in range(len(data_frame)):
        sentence = data_frame['Text'][row_num]

        sentiment = predict(sentence)

        # Calculate overall sentiment by compound score
        sentiment_list.append(sentiment)

    data_frame['Sentiment'] = sentiment_list

    return data_frame
# *** Backend operation
# Read comment csv data
# df = pd.read_csv('data/comment.csv')

# WSGI Application
# Provide template folder name
# The default folder name should be "templates" else need to mention custom folder name
app = Flask(__name__, template_folder='templateFiles')

app.secret_key = 'You Will Never Guess'


# @app.route('/')
# def welcome():
#     return "Welcome to great generalized sentiment analysis"

@app.route('/')
def index():
    return render_template('index_upload_and_show_data.html')


@app.route('/', methods=("POST", "GET"))
def uploadFile():
    if request.method == 'POST':
        uploaded_file = request.files['uploaded-file']
        df = pd.read_csv(uploaded_file)
        session['uploaded_csv_file'] = df.to_json()
        return render_template('index_upload_and_show_data_page2.html')


@app.route('/show_data')
def showData():
    # Get uploaded csv file from session as a json value
    uploaded_json = session.get('uploaded_csv_file', None)

    # Convert json to data frame
    uploaded_df = pd.DataFrame.from_dict(eval(uploaded_json))
    # Convert dataframe to html format
    uploaded_df_html = uploaded_df.to_html()
    return render_template('show_data.html', data=uploaded_df_html)


@app.route('/sentiment')
def SentimentAnalysis():
    # Get uploaded csv file from session as a json value
    uploaded_json = session.get('uploaded_csv_file', None)
    # Convert json to data frame
    uploaded_df = pd.DataFrame.from_dict(eval(uploaded_json))
    # Apply sentiment function to get sentiment score
    uploaded_df_sentiment = transformer(uploaded_df)
    uploaded_df_html = uploaded_df_sentiment.to_html()
    return render_template('show_data.html', data=uploaded_df_html)


if __name__ == '__main__':
    app.run(debug=True)
