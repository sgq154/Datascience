import pandas as p
import re
import nltk as n
from nltk.stem import PorterStemmer as ps
from nltk.corpus import stopwords as sw
from nltk.tokenize import word_tokenize as wt

from sklearn.model_selection import train_test_split as tts

#########################Task 1#########################

# loading the fakenewscorpus csv into a dataframe
link = "https://raw.githubusercontent.com/several27/FakeNewsCorpus/master/news_sample.csv"

df = p.read_csv(link)

# droping rows and removing duplicates
df.dropna(subset=["content"], inplace=True)

df.drop_duplicates(subset=["content"], inplace=True)

# function for text preprocessing
n.download(["punkt", "stopwords"])

stop_w = set(sw.words("english"))

ps = ps()


def process_text(txt):
    # cleaning the text using regex
    txt = txt.lower()
    txt = re.sub(r'\d+', '<NUM>', txt)
    txt = re.sub(r'http\S+', '<URL>', txt)
    txt = re.sub(r'\S+@\S+', '<EMAIL>', txt)
    txt = re.sub(r'[^\w\s]', '', txt)

    # tokenize text
    tkns = wt(txt)

    # remove stopwords and stem remaining tokens
    stmd_tkns = [ps.stem(tkn) for tkn in tkns if tkn not in stop_w]

    return stmd_tkns


#########################Task 2#########################

# apply preprocessing to content column
df['text_process'] = df['content'].apply(process_text)

# count occurrences of URLs, dates, and numbers in content
url_pat = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
date_pat = re.compile(
    r'\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|(?:Nov|Dec)(?:ember)?)\s+\d{1,2},\s+\d{4}\b')
num_pat = re.compile(r'\b\d+\b')


def count_pat_occ(txt, pat):
    return len(re.findall(pat, txt))


df['url_count'] = df['content'].apply(lambda x: count_pat_occ(x, url_pat))
df['date_count'] = df['content'].apply(lambda x: count_pat_occ(x, date_pat))
df['num_count'] = df['content'].apply(lambda x: count_pat_occ(x, num_pat))

# find 100 most common words in content
content_txt = " ".join(df["content"].str.lower())
tkns = wt(content_txt)
freq_dist = n.FreqDist(tkns)
top_100_words = freq_dist.most_common(100)

# find 100 most common words in preprocessed content
preprocessed_tkns = [tkn for sublist in df["text_process"] for tkn in sublist]
preprocessed_freq_dist = n.FreqDist(preprocessed_tkns)
preprocessed_top_100_words = preprocessed_freq_dist.most_common(100)

#########################Task 3#########################

# saving preprocessed dataframe to  a new CSV file
df.to_csv("news_file.csv", index=False)

# sample 10% of rows from preprocessed dataframe randomly
sampled_df = df.sample(frac=0.1, random_state=42)

#########################Task 4#########################

# split dataset into training (80%), validation (10%), and test (10%) sets
train_data, temp_data = tts(df, test_size=0.2, random_state=42)
validation_data, test_data = tts(temp_data, test_size=0.5, random_state=42)
