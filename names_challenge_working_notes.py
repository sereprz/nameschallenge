from __future__ import division
import random
import csv
import urllib2
import pandas as pd
import nltk
from nltk.corpus import wordnet as wn
from nltk.tokenize import RegexpTokenizer
from nltk.util import ngrams

url = 'https://raw.githubusercontent.com/Avaaz/names_challenge/master/datasets/CSV_Database_of_First_Names.csv'
webpage = urllib2.urlopen(url)
datareader = csv.reader(webpage)
first_name = []

for row in datareader:
    first_name.append(row[0])

# I'm not sure why but this same code doesn't work for this file
# I found someone that had the same problem on stackoverflow and they solved it
# re-creating the file
# (http://stackoverflow.com/questions/19482970/python-get-list-from-pandas-dataframe-column-headers)

# Also, pandas.read_csv works just fine so I'll just work aroud it for now

last_name = pd.read_csv('https://raw.githubusercontent.com/Avaaz/names_challenge/master/datasets/CSV_Database_of_Last_Names.csv', header = None)
last_name = list(last_name.iloc[:,0])

url = 'https://raw.githubusercontent.com/Avaaz/names_challenge/master/datasets/chicago_employees.csv'
webpage = urllib2.urlopen(url)
datareader = csv.reader(webpage)
trainingset = []

datareader.next()

# The names in this set are in the format 'Last_name, First_name', I am putting them in a list as 'First_name Last_Name'
# I'm going to use the chicago dataset as a training set to create a reasobly big set of possible 3-grams (assumption (probably not so strong): if the name passed to the function has too many 3-grams that can't be found in the training set then it's probably gibberish) and the first_name/last_name lists as lookups

for row in datareader:
    trainingset.append(' '.join(list(reversed(row[0].title().split(',  ')))))

## add Olympic athlets (include foreign names)
url = 'https://raw.githubusercontent.com/Avaaz/names_challenge/master/datasets/olympicathletes.csv'
webpage = urllib2.urlopen(url)
datareader = csv.reader(webpage)
datareader.next()

for row in datareader:
    name = row[0].decode('utf-8','ignore').encode()
    if name not in trainingset:
        trainingset.append(name)

random.shuffle(trainingset)
testset = trainingset[:int(len(trainingset)*0.05)]
trainingset = trainingset[int(len(trainingset)*0.05) + 1:]

print 'Training set:', len(trainingset)
print 'Test set:', len(testset)


threegrams_names = []

for name in trainingset:
    for ngram in ngrams(name, 3):
        threegrams_names.append(ngram)

fdist = nltk.FreqDist(threegrams_names)

tokenizer = RegexpTokenizer(r'\w+')

def is_real_name(full_name):
    """
    Given `full_name` as a string, returns the best guest as to whether it is real

    Consider the follow names fake:
    1. empty
    2. 3-grams can't be created
    3. > 1/4 of 3-grams are gibberish
    4. contains non proper noun words

    Otherwise consider the name real
    """

    # split the full name into tokens
    tokens = tokenizer.tokenize(full_name)

    # 1. consider empty names fake
    if not tokens:
        return False

    # 2. if 3-grams can't be created consider fake
    # ngrams is a generator so if list(ngrams(full_name, 3)) = [] the name has less than 3 letters
    if not list(ngrams(full_name, 3)):
        return False

    # 3. reject gibberish
    # calculate a gibberish score >= 0
    # for every ngram not in the training set, add 1
    # if more than a quarter of ngrams are gibberish, consider fake
    gibberish_score = 0

    # generate 3-grams from full_name. for each 3-gram
    for gram in ngrams(full_name,3):
        # chek if it appears in the training set
        if gram not in fdist.keys():
            # increment the gibberish score by one
            gibberish_score += 1

    if gibberish_score/len(list(ngrams(full_name,3))) > 0.25:
        return False

    # 4.
    # check if it's a sentence/a series of words that are not names, e.g. 'Some Name'
    for token in tokens:
        # ignore len(token) <= 2: single letters are actually allowed in names, as well as 2-letters words (e.g. JR)
        if len(token) > 2:
            # wn (wordnet) is a semantic dictionary for the english language, if a word appears in it,
            if wn.lemmas(token):
                # and it's not on the lists of possible names/last names
                if not (token in first_name or token in last_name):
                    # consider it fake
                    return False

    return True


#### Test

for name in testset:
    if not is_real_name(name):
        print name, is_real_name(name)

not_true_name = []
for name in testset:
    if not is_real_name(name):
        not_true_name.append(name)

print 'Percentage of real names misclassified as fake:', len(not_true_name)/len(testset)
