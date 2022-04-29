# Implementing Naive Bayes Classifier for Sentiment Analysis

# importing required libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

print('\nImporting necessary files and libraries...\n')

# importing data from the CSV file
data = pd.read_csv(r'../Data/data.csv')

print(f'\nThe data from the CSV file \n{data.head()}\n')

# Dividing Data for Training and Testing

# splitting data for training and testing purpose i.e., 80% and 20% respectively
X = data[['selected_text', 'sentiment']]
train_data, test_data = train_test_split(X, test_size=0.2, random_state=75)
print(f'\nThe sample of data to be trained \n{train_data.head()}\n')

print(f'\nThe sample of data to be tested \n{test_data.head()}\n')

# Classifying Data According to the Sentiments

# storing data according to their labels
neutral_data = train_data.loc[train_data['sentiment'].__eq__('neutral')]
positive_data = train_data.loc[train_data['sentiment'].__eq__('positive')]
negative_data = train_data.loc[train_data['sentiment'].__eq__('negative')]

# Preprocessing the Data

# Removing Punctuations and Stop Words

# function for replacing unnecessary punctuations and stop words
stop_words = set(stopwords.words('english'))
punctuations = ['.', ',', '!', '?', '"', '#', '+', '<', '>', '@', '(', ')']


def remove_punctuations_stopwords(text: str) -> list:
    # to remove punctuations
    for symbol in punctuations:
        text = text.lower().replace(symbol, "")

    # to remove stop words and converting into list of words
    text = [each_word for each_word in text.split() if each_word not in stop_words]

    return text


# Stemming the Sentences
def stem_words(text: list) -> list:
    ps = PorterStemmer()

    # stemming each word from the list of text
    stemmed_words = [ps.stem(each_word) for each_word in text]

    return stemmed_words


def preprocess_text(text: str) -> list:
    # remove punctuations and stop words
    text = remove_punctuations_stopwords(text)

    # stemming the list of words
    stemmed_words = stem_words(text)

    return stemmed_words


# Training

# Creating Bag of Words and List of Words from Given Sentences

bag_of_words = []
neutral_sentences = []
positive_sentences = []
negative_sentences = []


def create_vocabulary(classified_data: pd):
    for text in classified_data['selected_text'].astype(str):  # type casting pandas object into string type

        bag_of_words.extend(
            preprocess_text(text))  # adding words into bag of words by removing punctuations and stop words

        if classified_data['sentiment'].unique().__eq__(
                ['neutral']):  # checking if the classified data are labelled neutral

            neutral_sentences.append(preprocess_text(text))  # adding each sentence as list to sentences

        elif classified_data['sentiment'].unique().__eq__(['positive']):

            positive_sentences.append(preprocess_text(text))

        else:

            negative_sentences.append(preprocess_text(text))


print('\nCreating the Bag of Words......\n')

# calling method for creating bag of words by passing classified data as parameter
create_vocabulary(neutral_data)

create_vocabulary(positive_data)

create_vocabulary(negative_data)

# for removing duplicate words in bag of words
bag_of_words = np.array(list(set(bag_of_words)))

print(f'\nThe length of the Bag of Words is {len(bag_of_words)}\n')


# Creating Text Vector (Counting the frequency of words in a sentence)

# for counting the number of words presented in the sentence as that of bag of words
def create_text_vector(sentences: list) -> list:
    word_counts = []

    for sentence in sentences:
        word_count = []
        for word in bag_of_words:
            word_count.append(
                sentence.count(word))  # counting and adding words presented in the sentence from bag of words

        word_counts.append(word_count)  # adding total word counts for each sentence into the list

    return word_counts


print('\nCreating text vectors, \nThis may take few minutes....\n')

# calling methods for creating text vector by passing list of specific labelled sentences as parameter
neutral = np.array(create_text_vector(neutral_sentences))
print('Completed for Neutral sentences')

positive = np.array(create_text_vector(positive_sentences))
print('Completed for Positive sentences')

negative = np.array(create_text_vector(negative_sentences))
print('Completed for Negative sentences')

# Calculating Conditional Probabilities Using Improvised Formulae

print(f'\nCalculating Probabilities....\n')

# calculating probabilities for each word using Naive Bayes classifier
neutral_probabilities = (np.sum(neutral, axis=0) + 1) / (np.sum(neutral) + len(bag_of_words))

positive_probabilities = (np.sum(positive, axis=0) + 1) / (np.sum(positive) + len(bag_of_words))

negative_probabilities = (np.sum(negative, axis=0) + 1) / (np.sum(negative) + len(bag_of_words))

# probability of specific sentiments from the entire training dataset
neutral_probability = len(neutral_data) / len(train_data)
positive_probability = len(positive_data) / len(train_data)
negative_probability = len(negative_data) / len(train_data)


# Testing

# Predicting Sentiments of Input Data for Testing

# for predicting inputs
def predict(sentence: str) -> str:
    # calculate total probability for each type of sentiment
    gross_neutral = neutral_probability
    gross_positive = positive_probability
    gross_negative = negative_probability

    refined_words = preprocess_text(sentence)

    # traversing through list of words for calculating gross probability for every sentiments
    for word in refined_words:
        if word in bag_of_words:  # checking if the word exists in the vocabulary
            gross_neutral *= neutral_probabilities[np.where(bag_of_words.__eq__(word))[0][
                0]]  # getting and multiplying the probability of the word using index
            gross_positive *= positive_probabilities[np.where(bag_of_words.__eq__(word))[0][0]]
            gross_negative *= negative_probabilities[np.where(bag_of_words.__eq__(word))[0][0]]

    if gross_neutral > gross_positive and gross_neutral > gross_negative:
        return 'neutral'
    elif gross_positive > gross_neutral and gross_positive > gross_negative:
        return 'positive'
    else:
        return 'negative'


# Calculating Accuracy Score of Developed Model using The Test Data

# function for calculating the accuracy
def calculate_accuracy(raw_data: pd) -> str:
    accuracy_score = 0
    for each in range(len(raw_data)):
        row = raw_data.iloc[each]  # each row of dataframe via index

        if predict(row['selected_text']).__eq__(row['sentiment']):  # comparing sentiment of input and output text
            accuracy_score += 1

    return f"The accuracy of the model is {round(accuracy_score / len(raw_data) * 100, 2)}%"


print('\nCalculating the accuracy of the model using test data...... \nThis may take few seconds....\n')

print(f'\n{calculate_accuracy(test_data)}\n')


# Predicting Sentiment of the New Input
def new_input() -> str:
    text_to_classify = input('\nEnter a sentence to classify the sentiment: ')

    if text_to_classify != "":
        return f"The Sentiment of the input is: '{predict(text_to_classify)}'"
    else:
        return "Please! Provide some text."


flag = True
while flag:
    print(f'\n{new_input()}')
    continue_ = input('\nWant to Test new Sentence?[Y/N]: ')

    if continue_ in ['N', 'n']:
        flag = False
