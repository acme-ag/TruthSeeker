[![truthSeeket](https://i.postimg.cc/wB3n43J9/temp-Imagec2-Nze9.avif)](https://postimg.cc/cKyDrxXz)
# TruthSeeker
Traditional ML approach to fake/real news classification based on TruthSeeker dataset

## Contents:
1. Brief overview
2. Features explanarion
3. 1 - Data perprocessing
4. 2 - Explaratory Data Analysis
5. 3 - Modelling
6. Inferences and Summary

## Overview

The data originates from the Canadian Institute of Cybersecurity [University of New Brunswick](https://www.unb.ca/cic/about/index.html) under the title "CIC Truth Seeker Dataset 2023". The dataset encompasses over 100,000 tweets related to 700 real and 700 fake news stories (statements) spanning from 2009 to 2022. These texts are categorized into some featires like text features, lexical and meta-data features, making it one of the largest ground truth datasets ever created for detecting fake news on Twitter. 

The dataset includes two .csv files (**Features_For_Traditional_ML_Techniques.csv** -- for "traditional" ML, that's the file we'll be using; and **Truth_Seeker_Model_Dataset.csv** for NLP) and .xls file with timestamps. We're gonna work on traditional ML techniques so we'll use that dataset that contains 134,199 rows and 64 columns. Our target here is BinaryNumTarget (or majority_target in boolean representation) with values 1 (true) or 0 (fake). And we want, based on the features in the dataset, that are basically the codified tweet texts, make best possible models that would categorize the statements (news headlines) if they are true or fake.

## Features explanation

Here's the brief features description (with some of my explanations):

Some of the features are self-explanatory, some of them need to be decribed.

1.	majority_target - Truth value of the statement (boolean)
2.	BinaryNumTarget - Binary representation of the statement's truth value (1 = True / 0 = False)
3.	BotScore – continuous metric. Synthetic metric [0…1] to assess if a user bot or not based on different parameters. See paper page 11, 9.1
4.	BotScoreBinary - Binary representation.
5.	statement - Headline of a new article, text. 
6.	tweet - twitter posts related to the associated manual keywords (see dataset for NLP), text
7.	unique_count - number of unique, complex words. No context from research team what they mean by that. Unique words in text?
8.	Adpositions -- Includes both prepositions (in, on, at, by, with…) and postpositions.
Fake news may use fewer adpositions to keep text simpler.
9.	Adverbs -- Quickly, slowly, yesterday, last week, here, there, today, daily, never, rarely, extremely, annually
10.	Embeddings -- Word embeddings are numerical representations of words in a continuous vector space, where words with similar meanings are located closer to each other. These embeddings are used to capture semantic relationships between words. Each word in the text is converted into a high-dimensional vector.
11.	following
12.	Word count – all words that can be recognized as lexical units
13.	present_verb - number of present tense verbs.
14.	followers_count - number of followers.
15.	total_count – all text units, words, including unrecognized combinations of characters, numbers and signs like that ‘–‘
16.	past_verb number of past tense verbs.
17.	friends_count - number of friends.
18.	adjectives - number of adjectives.
19.	favourites_count - number of favorites across all tweets. This metric measures how many users have expressed a positive reaction to the content by favoriting it.
20.	pronouns - number of pronouns.
21.	statuses_count - number of tweets.
22.	TO’s - number of “to” usages.
23.	listed_count - number of tweets the user has in lists.
24.	determiners - number of determiners.
Articles: a/an/the
pointers: this, that, these, those
Ownership, posessives: my, your, his, her, its, our, their
Quantifiers: some, many, few, several, all, every
25.	mentions - number of times the user was mentioned.
26.	conjunctions - number of conjunctions.
27.	quotes - number of times the user has been quote tweeted.
28.	dots - number of (.) used.
29.	replies - number of replies the user has.
30.	exclamations - number of (!) used.
31.	retweets - number of retweets the user has.
32.	question - number of (?) used.
33.	favourites - number of favourites the user has.
34.	ampersand - number of (&) used. 
35.	hashtags number of hashtags (#) the user has used.
36.	capitals - Number of capitalized letters.
37.	URLs - number of URLs the user has posted.
38.	quotes - number of quotation marks ("") used.
39.	digits - number of digits (0-9) used.
40.	cred - credibility score. Synthetic score [0…1] to assess user’s credibility. See the paper page 13, 9.3
41.	long_word_freq - number of long words.
42.	normalized_influence - influence score the user has, normalized. Synthetic metric [0…1] made by authors to assess a user’s influence (normilized) ¬– how user’s actions affect other users. See page 11.
43.	Max Word - length of the longest word in the sentence.
44.	Min Word - length of the shortest word in the sentence.
45.	Avg Word Length - average length of words in the sentence.
46.	ORG_percent - Percent of text including spaCy ORG tags. Percentage of the text that contains named entities tagged as organizations by the spaCy NLP library. The ORG tag is used by spaCy to identify named entities that are organizations, such as companies, agencies, institutions, etc.
47.	NORP_percent - Percent of text including spaCy NORP tags. NORP stands for nationalities or religious or political groups.
48.	PERSON_percent - Percent of text including spaCy PERSON tags.
49.	MONEY_percent - Percent of text including spaCy MONEY tags. MONEY tags are used to identify mentions of monetary values, amounts, and currency symbols in the text.
50.	DATA_percent - Percent of text including spaCy DATA tags. mentions of data points, numerical information, statistics, or specific datasets within the text
51.	GPE_percent - Percent of text including spaCy GPE tags. GPE stands for Geopolitical Entities.
52.	CARDINAL_percent - Percent of text including spaCy CARDINAL tags. These are numbers that indicate quantity but not order, such as “one,” “two,” “100,” etc
53.	PERCENT_percent - Percent of text including spaCy PERCENT tags.
54.	ORDINAL_percent - Percent of text including spaCy ORDINAL tags. See CARDINAL
55.	FAC_percent - Percent of text including spaCy FAC tags. This tag is used to identify mentions of buildings, airports, highways, bridges, etc., in the text.
56.	LAW_percent - Percent of text including spaCy LAW tags. Law documents. This can include specific laws, legal statutes, regulations, treaties, and other formal legal documents.
57.	PRODUCT_percent - Percent of text including spaCy PRODUCT tags. Mentions of products. This can include consumer products, services, technologies, brands, and other commercial items.
58.	EVENT_percent - Percent of text including spaCy EVENT tags. Mentions of events -- occurrences such as concerts, hurricanes, sports events, wars, and other significant happenings.
59.	TIME_percent - Percent of text including spaCy TIME tags. Referes to time entities.
60.	short_word_freq - number of short words.
61.	LOC_percent - Percent of text including spaCy LOC tags. Locations mentions
62.	WORK_OF_ART_percent - Percent of text including spaCy WOA tags. WORK_OF_ART tags are to identify mentions of creative works: titles of books, songs, movies, paintings, and other artistic creations.
63.	QUANTITY_percent - Percent of text including spaCy QUANTITY tags. measurements, amounts, distances, weights, and other numerical expressions that indicate a quantity.
64.	LANGUAGE_percent - Percent of text including spaCy LANGUAGE tag. Mentiojns of languages like English, Spanish, French…

## Work stages

Since the project is rather long-lasting I divided it into 3 parts: preprocessing, EDA and modelling itself.

### 1. Data perprocessing 

In this section, I check the dataset for data validity and perform some initial transformations. For example, several variables were in int64 and float64 formats, which was somewhat excessive. Consequently, I managed to reduce the dataset size by almost three times in megabytes. This is not as important when working with small datasets like this one, but it can be crucial for larger volumes. Next, a preliminary analysis of data consistency revealed discrepancies with the specified parameters. That is, you look at the text, compare it with its parameters in the variables, and notice inconsistencies. To address this, I recalculated almost all the parameters. Spoiler: this only slightly improved the final results. However, it was interesting and useful practice.

### 2. Explaratory Data Analysis

In the Data Analysis section, I examine the data for distribution, right or left-skewed data, deviations, and correlations. In this section, I also perform feature engineering and – although it’s not as significant for the actual analysis, it was interesting – I examine the distribution of the vocabulary across different slices. What words are used in discussions of fake news and real news, and how the vocabulary is distributed across topics.

### 3. Modelling

In the Modelling section, I create synthetic new variables, deal with outliers, and conduct PCA analysis to determine the number of variables sufficient for creating robust models. Ultimately, I create several models: logistic regression, which serves as our baseline model, Support Vector Machines with two kernels: RBF and Polynomial, Deep Neural Networks, Random Forest, XGBoost, and compare the results. Since the target class is balanced, the main metric is the accuracy score.

## In conclusion

The research team, in their paper on this dataset, emphasizes that their work was focused on NLP models like BERT and others, and the dataset for traditional machine learning techniques was created more as a complement. NLP techniques, when applied to this dataset, show better results for determining fake and real news, as confirmed by OpenAI researchers who applied their algorithms to this dataset. However, I found it interesting to apply traditional techniques. I believe this is the best result that can be achieved using them with this dataset.


| Model                   | Train accuracy | Test accuracy |
|-------------------------|----------------|---------------|
| LogisticRegression      | 0.6552         | 0.6491        |
| SVC (RBF)               | 0.7980         | 0.7179        |
| SVC (Poly)              | 0.7700         | 0.6974        |
| RandomForestClassifier  | 0.7480         | 0.6801        |
| XGBClassifier           | 0.8360         | **0.7336**        |
| Sequential              | 0.7382         | 0.7058        |
