import pandas as pd
import string
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.neural_network import MLPClassifier

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.normalization import BatchNormalization

def clean_text(text):
	text = text.lower()
	text = re.sub('\[.*?\]', '', text)
	text = re.sub('https?://\S+|www\.\S+', '', text)
	text = re.sub('<.*?>+', '', text)
	text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
	text = re.sub('\n', '', text)
	text = re.sub('\w*\d\w*', '', text)
	return text

def data_preprocessing(data_path='../data/fake_job_postings.csv'):
	df = pd.read_csv(data_path)
	# print(df.isnull().sum())

	df.drop(['job_id', 'location', 'department', 'salary_range'], \
		axis = 1, inplace = True)

	# def convert_salary_range_to_avg():
	# 	avg_salary = []
	# 	for salary_range in df['salary_range'].tolist():
	# 		try:
	# 			[min_val, max_val] = salary_range.split('-')
	# 			avg_salary.append((int(min_val)+int(max_val))/2)
	# 		except:
	# 			avg_salary.append(nan)
	# 	df['salary_range'] = avg_salary
	# 	df['salary_range'] = df['salary_range'].bfill(axis=0)
	text_headers = ['title', 'company_profile', 'description', 'requirements', 'benefits']
	df[text_headers] = df[text_headers].fillna('')

	categorical_headers = ['employment_type', 'required_experience', 'required_education', 'industry', 'function']
	df[categorical_headers] = df[categorical_headers].bfill()

	df = df.dropna(axis = 0, how = 'any')
	df['description'] = df['description'] + ' ' + df['title'] + ' ' + df['company_profile']\
	 + ' ' + df['requirements'] + ' ' + df['benefits']
	df.drop(['title', 'company_profile', 'requirements', 'benefits'], axis = 1, inplace = True)
	
	cleaned_description = []
	for text in df['description'].tolist():
		cleaned_description.append(clean_text(text))
	df['description'] = cleaned_description
	######
	tfidf_vectorizer = TfidfVectorizer(min_df=3, stop_words=stopwords.words('english'))
	df['description'] = tfidf_vectorizer.fit_transform(df['description']).toarray()

	for header in categorical_headers:
		df[header] = df[header].factorize()[0]

	return df

def MultinomialNB_classifier(df):
	X = df.drop(columns=['fraudulent'], axis=1)
	y = df['fraudulent']
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
	model = MultinomialNB()
	model.fit(X_train, y_train)
	preds = model.predict(X_test)
	auc = roc_auc_score(y_test, preds)
	print('With MultinomialNB_classifier, AUC score = ', auc)

def GaussianNB_classifier(df):
	X = df.drop(columns=['fraudulent'], axis=1)
	y = df['fraudulent']
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
	model = GaussianNB()
	model.fit(X_train, y_train)
	preds = model.predict(X_test)
	auc = roc_auc_score(y_test, preds)
	print('With GaussianNB_classifier, AUC score = ', auc)

def MLP_classifier(df):
	X = df.drop(columns=['fraudulent'], axis=1)
	y = df['fraudulent']
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
	model = MLPClassifier(solver = 'adam', activation = 'relu', hidden_layer_sizes = (100, 50, 30), max_iter = 1000)
	model.fit(X_train, y_train)
	preds = model.predict(X_test)
	auc = roc_auc_score(y_test, preds)
	print('With MLP_classifier, AUC score = ', auc)

def seq_network(df):
	X = df.drop(columns=['fraudulent'], axis=1)
	y = df['fraudulent']
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

	model = Sequential()

	model.add(Dense(200, input_dim=200, activation='relu'))
	model.add(Dropout(0.2))
	model.add(BatchNormalization())

	model.add(Dense(100, activation='relu'))
	model.add(Dropout(0.2))
	model.add(BatchNormalization())

	model.add(Dense(100, activation='relu'))
	model.add(Dropout(0.2))
	model.add(BatchNormalization())

	model.add(Dense(1))
	model.add(Activation('sigmoid'))

	# compile the model
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

	model.fit(X_train, y=y_train, batch_size=64, 
		  epochs=10, verbose=1, 
		  validation_data=(X_test, y_test))

	preds = model.predict(X_test)
	print(preds)
	auc = roc_auc_score(y_test, preds)
	print('With MLP_classifier, AUC score = ', auc)


def main():
	df = data_preprocessing()
	# MultinomialNB_classifier(df)
	# GaussianNB_classifier(df)
	# MLP_classifier(df)
	seq_network(df)

if __name__ == '__main__':
	main()
