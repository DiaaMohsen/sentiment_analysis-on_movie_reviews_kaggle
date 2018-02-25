
Sentiment Analysis on Movie Reviews
Classify the sentiment of sentences from the Rotten Tomatoes dataset on kaggle

Kaggle Competition: https://www.kaggle.com/c/sentiment-analysis-on-movie-reviews

-On test.tsv attached in competition, i got overall accuracy 0.61835

- I got overall accuracy: 0.686616258704 on testset extracted from splitting train.tsv
                   precision    recall  f1-score   support
         NEGATIVE       0.78      0.54      0.63      3011
SOMEWHAT NEGATIVE       0.48      0.61      0.54      6466
          NEUTRAL       0.81      0.75      0.78     25868
SOMEWHAT POSITIVE       0.52      0.66      0.58      7755
         POSITIVE       0.79      0.58      0.67      3718

      avg / total       0.71      0.69      0.69     46818

I built 3 classifers using [LinearSVC with CountVectorizer] as follows:
- Classifier-1: classify {NEG, NEU, POS} on normalized sentiment values: {-1, 0, 1}
- Classifier-2: classify {NEG, SMW NEG} on sentiment values: {0, 1}
- Classifier-3: classify {SMW POS, POS} on sentiment values: {4, 5}

How does it work?
- classify data with classifier-1
	if data labeled with -1:
		classify it with classifier-2
	if data labeled with 1:
		classify it with classifier-3
	if data labeled with 0:
		no thing to run


Which models i used?
- LinearSVC got the best results
- SGDClassifier 
- MultinomialNB
- TfidfVectorizer

ToDO:
- Downsampling
- Tuning models more than what i did
- GridSearch
- Try more models or combination of models