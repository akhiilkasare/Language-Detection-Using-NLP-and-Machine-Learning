# Language Detection Using NLP & Machine Learning

![Python 3.6](https://img.shields.io/badge/Python-3.6-brightgreen.svg) ![scikit-learnn](https://img.shields.io/badge/Library-Scikit_Learn-orange.svg) ![NLP](https://img.shields.io/badge/Library-NLP-yellow.svg)


**Language Detection** using the European Parliament Proceedings Parallel Corpus. European Parliament Proceedings Parallel Corpus is a text dataset used for evaluating language detection engines. The 1.5GB corpus includes 21 languages spoken in EU. This project aims to build a machine learning model trained on this dataset to predict new unseen data. The Language Detection falls basically into Text Classification part.

- List of all the languages whose detection is supported:
   - 'bg': Bulgarian
   - 'cs': Czech
   - 'da': Danish
   - 'de': German
   - 'el': Greek, Modern
   - 'en': English
   - 'es': Spanish
   - 'et': Estonian
   - 'fi': Finnish
   - 'fr': French
   - 'hu': Hungarian
   - 'it': Italian
   - 'lt': Lithuanian
   - 'lv': Latvian
   - 'nl': Dutch
   - 'pl': Polish
   - 'pt': Portuguese
   - 'ro': Romanian
   - 'sk': Slovak
   - 'sl': Slovenian
   - 'sv': Swedish

### Table of content : 

    1. Importing all the essiantial libraries.
    2. Loading the data.
    3. Data Preprocessing.
    4. Transforming the data into a single dataset.
    5. Dividing the dataset.
    6. Converting the words into vectors. 
    7. Creating a pipeline
    8. Prediction and Model evaluation
    9. Creating a pickle file
        
# 1. Importing all the essential libraries

```
import string 
import re
import codecs
import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import feature_extraction
from sklearn import linear_model
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import pipeline
from sklearn.model_selection import train_test_split
from sklearn import metrics
import itertools
```

# 2. Loading the dataset

Using the pandas library we were able to read the dataset of the respective languages

#### Reading the data for English dataset.
```
english_df = pd.read_csv("europarl-v7.bg-en.en", "ut-8", header=None, names=["English"])
```
#### Reading the data for German dataset.
````
german_df = pd.read_csv("europarl-v7.de-en.de", "utf-8", header=None, names=["German"])
````

#### Reading the dataset for French dataset.
```
french_df = pd.read_csv("europarl-v7.fr-en.fr", "utf-8", header=None, names=["French"])
```

#### Reading the dataset for Spanish dataset.
```
spanish_df = pd.read_csv("europarl-v7.es-en.es", "utf-8", header=None, names=["Spanish"])
```
And many other languages can be read similarly

# 3. Data Text Pre-Processing.

Before giving our model to our machine learning model we need to pre process the data. The main aim of data pre processing is remove the unwanted characters, punctuations and many other noisy data the list of the noisy data is
```
for char in string.punctuation:
    print(char, end = ' ')
translate_table = dict((ord(char), None) for char in string.punctuation)

Output: 
! " # $ % & ' ( ) * + , - . / : ; < = > ? @ [ \ ] ^ _ ` { | } ~ 
```
Text Pre-Processing basically involves.
- **Tokenization** : Splitting the sentences into words are basically called as tokenization. The words here are called tokens.

- **Stopwords removal** : Stopwords are nothing but these are frequently occuring words and it does not add any information to the model. For example : is,the,prepositions and article words.
- **Lower case conversion** : The necessity of converting all the words into lower case is that (for ex: Mumbai,MUMBAI,mumbai) the meaning of the word is same but our model will detect them as 3 different words. To avoid this confusion we convert all the sentences and words into lower case.
- **Removing numeric / digits** : Having numbers into our sentences dosent make any sence to our model it will just add up the number of features 
- **Removing Punctuation / Special characters** : Same as numeric values it dosent add any value to our model.
- **Removing characters for foreign languages** : If you want to identify the languages like chinese and japanese and you want to make your model to understand these languages in this case you can also remove the character(only for the foreign languages).
- **Normalization** : The words can be be written into different formats (USA, U.S.A, usa) we need to convert these into one format. 
- **Stemming & Lemmatization** : This is the most important step to perform. In this we are cutting the words either we are bringing the word to its original /root form or we are cutting the word into it form and int from.

# 4. Transforming the dataset into single dataset.

We have now pre processed all the language dataset and now we need to just combine all the individual dataset into a single Data frame for our convinience.

# 5. Dividing the dataset. 

After we have combined all the individual dataset we need to split our dataset Independent(x) and Dependent(y) variable(target variable) for prediction.

# 6. Converting the words into vectors.
There are various methods for converting the words into vectors. Model cannot understand raw form we need to convert into something called as 0's and 1's. The most commonly used vectorization techniques are as follows : 
- **Bag-of-Words** (Count Vectorizer) :
Bag of words converts text into set of vectors containing the count of word occurrences in the document.

- **TF-IDF** :
TF-IDF creates vectors from the text which contains information on the more important words and the less important ones as well.

- **Word2Vec** :
Word2Vec creates vectors that are numerical representation of word features, features such as the context of individual words. The purpose and usefulness of Word2Vec is to group the vectors of similar words together into vector space. That is it detects similarities mathematically. 

# 7. Creating a pipeline.
Using the vectorizer and fitting the model into the pipeline.

# 8. Prediction and model evaluation.
In this project I've used Logistic Regression it has performed well with all this data and was able to acheive 92% accuracy.

# 9. Creating a pickle file.
Creating a pickle file using .pickle file and can be used for the deployment of the model over web.















