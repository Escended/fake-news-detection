# text preprocessing imports
import nltk
import string

from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize


def remove_html(text):
    soup = BeautifulSoup(text)
    html_free = soup.get_text()
    return html_free


# remove punctuation that we dont want to have tokenized
def remove_punctuation(text):
    no_punctuation = "".join([c for c in text if c not in string.punctuation])
    return no_punctuation


# tokenize each row using regex tokenizer, creating a list of words at every white space '\w'
tokenizer = RegexpTokenizer(r'\w+')


# remove stopwords such as 'this' and 'in' as these would be the most common words when extracting features
def remove_stopwords(text):
    words = [w for w in text if w not in stopwords.words('english')]
    return words


stemmer = PorterStemmer()


def word_stemmer(text):
    stem_text = " ".join([stemmer.stem(i) for i in text])
    return stem_text


# group all the functions into one function call that we can use on individual paragraphs later and to preprocess the
# dataset
def preprocess_text(x):
    x = remove_html(x)
    x = remove_punctuation(x)
    x = tokenizer.tokenize(x.lower())
    x = remove_stopwords(x)
    x = word_stemmer(x)
    return x
