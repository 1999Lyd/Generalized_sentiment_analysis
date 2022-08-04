# Generalized_sentiment_analysis
[![build](https://github.com/1999Lyd/Generalized_sentiment_analysis/actions/workflows/main.yml/badge.svg)](https://github.com/1999Lyd/Generalized_sentiment_analysis/actions/workflows/main.yml)
## Introduction
- A generalized sentiment analysis tool based the pre-trained XLNet provided by Huggingface transformers, which is fine-tuned by data from amazon food reviews, twitters and Airline reviews in order to achieve the generalized sentiment analysis purpose.
## Get started
- Download model from [GCP cloud storage](https://storage.googleapis.com/lyd990404.appspot.com/pretrained1.pt) to the 'models' folder
- Install required package
``` pip install -r requirements.txt```
- run service locally``` python app.py```
- or run in container
``` 
    docker build . -t app.py
    docker run -p 8080:8080 app.py 
```
- [web service link](https://tzzyy9utnn.us-east-1.awsapprunner.com )
