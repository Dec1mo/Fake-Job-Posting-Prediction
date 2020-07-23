import pandas as pd
import re
from string import punctuation
from scipy.sparse import csr_matrix, hstack
import pickle
from sklearn import svm
import nltk 
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

stemmer = PorterStemmer()

def clean_text(text):
	text = text.lower()
	text = re.sub('\[.*?\]', ' ', text)
	text = re.sub('https?://\S+|www\.\S+', ' ', text)
	text = re.sub('<.*?>+', ' ', text)
	text = re.sub('[%s]' % re.escape(punctuation), ' ', text)
	text = re.sub('\w*\d\w*', ' ', text)

	return ' '.join([stemmer.stem(w) for w in word_tokenize(text)])

def data_preprocessing(data_path='../data/fake_job_postings.csv'):
	df = pd.read_csv(data_path)
	# print(df.isnull().sum())

	df.drop(['job_id'], \
		axis = 1, inplace = True)
	

	text_headers = ['title', 'company_profile', 'description', 'requirements', 'benefits']
	df[text_headers] = df[text_headers].fillna('')

	categorical_headers = ['department', 'salary_range',
	                         'telecommuting', 'has_company_logo', 
	                         'has_questions', 'employment_type', 
	                         'required_experience', 'required_education',
	                         'industry', 'function']
	df[categorical_headers] = df[categorical_headers].fillna('None')

	# df = df.fillna('None')
	# df = df.dropna(axis = 0, how = 'any')

	# Handle text-typed data
	df['description'] = df['description'] + ' ' + df['title'] + ' ' + df['company_profile']\
	 + ' ' + df['requirements'] + ' ' + df['benefits']
	# df.drop(['title', 'company_profile', 'requirements', 'benefits'], axis = 1, inplace = True)
	
	cleaned_description = []
	for text in df['description'].tolist():
		cleaned_description.append(clean_text(text))
	# df['description'] = cleaned_description
	
	# vectorizer = CountVectorizer(min_df=3, stop_words='english')
	vectorizer = TfidfVectorizer(lowercase=False, stop_words='english')
	X_text = vectorizer.fit_transform(cleaned_description)

	# Handle categorical-typed data
	data_dict = df[categorical_headers].to_dict('records')
	dict_vectorizer = DictVectorizer()
	X_cate = dict_vectorizer.fit_transform(data_dict)

	X = hstack([X_text, X_cate], format='csr')
	print(X.shape)
	y = df['fraudulent']
	return X, y

def MLP_classifier(X_train, X_test, y_train, y_test):
	model = MLPClassifier(solver = 'adam', activation = 'relu', hidden_layer_sizes = (100, 50, 30), max_iter = 2000)
	# model.fit(X_train, y_train)
	# pickle.dump(model, open('os_MLPClassifier.pkl', 'wb'))
	model = pickle.load(open('MLPClassifier.pkl', 'rb'))
	y_pred = model.predict(X_test)
	auc = roc_auc_score(y_test, y_pred)
	print('With MLP_classifier, AUC score = ', auc)
	print(classification_report(y_test, y_pred))


def LR_classifier(X_train, X_test, y_train, y_test):
	model = LogisticRegression(penalty='l2', random_state=42, 
		solver='lbfgs', max_iter=100000)
	# model.fit(X_train, y_train)
	# pickle.dump(model, open('os_LogisticRegression.pkl', 'wb'))
	model = pickle.load(open('LogisticRegression.pkl', 'rb'))
	y_pred = model.predict(X_test)
	auc = roc_auc_score(y_test, y_pred)
	print('With LR_classifier, AUC score = ', auc)
	print(classification_report(y_test, y_pred))

def main():
	# X, y = data_preprocessing()
	# pickle.dump((X, y), open('text_X_y.pkl', 'wb'))

	# MLP_classifier(X, y) 
	# With MLP_classifier, AUC score =  0.893525917524474 (TfidfVectorizer) and 0.8828455717665389 (CountVectorizer)
	# LR_classifier(X, y) 
	# With LR_classifier, AUC score =  0.7861209923642258 (TfidfVectorizer) and 0.879139385815676 (CountVectorizer)


	X, y = pickle.load(open('X_y.pkl', 'rb'))
	X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=1)
	# print(y_test.value_counts())
	os_X_train, os_y_train = SMOTE().fit_resample(X_train, y_train)

	print(y_test.value_counts())
	MLP_classifier(os_X_train, X_test, os_y_train, y_test) 
	LR_classifier(os_X_train, X_test, os_y_train, y_test) 

	# print(y_test.value_counts())
	# us_X_train, us_y_train = RandomUnderSampler().fit_resample(X_train, y_train)
	# print(y_test.value_counts())
	# MLP_classifier(us_X_train, X_test, us_y_train, y_test) 
	# LR_classifier(us_X_train, X_test, us_y_train, y_test) 

if __name__ == '__main__':
	main()
