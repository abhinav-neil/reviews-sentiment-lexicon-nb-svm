# Sentiment Detection in Movie Reviews using Lexicon, Naive-Bayes & SVM

## Introduction
Perform sentiment detection on movie reviews with different classification techniques. The [dataset](https://gist.githubusercontent.com/bastings/d47423301cca214e3930061a5a75e177/raw/5113687382919e22b1f09ce71a8fecd1687a5760/reviews.json) contains 1000 positive & 1000 negative movie reviews. Each review is a and consists of one or more sentences. The data contains the text of the reviews, where each document consists
of the sentences in the review, the sentiment of the review and an index
(cv) for cross-validation. The techniques used are derived from the paper:
>   Bo Pang, Lillian Lee, and Shivakumar Vaithyanathan
(2002). 
[Thumbs up? Sentiment Classification using Machine Learning
Techniques](https://dl.acm.org/citation.cfm?id=1118704). EMNLP.

## 1. Lexicon-based approach
A traditional approach to classify documents according to their sentiment is the lexicon-based approach. To implement this approach, you need a **sentiment lexicon**, i.e., a list of words annotated with a sentiment label (e.g., positive and negative, or a score from 0 to 5).

We use the sentiment lexicon released by Wilson et al. (2005).

> Theresa Wilson, Janyce Wiebe, and Paul Hoffmann
(2005). [Recognizing Contextual Polarity in Phrase-Level Sentiment
Analysis](http://www.aclweb.org/anthology/H/H05/H05-1044.pdf). HLT-EMNLP.

[//]: # (The field *word1* contains the lemma, *priorpolarity* contains the sentiment label (positive, negative, both, or neutral), *type* gives you the magnitude of the word's sentiment (strong or weak), and *pos1* gives you the part-of-speech tag of the lemma. Some lemmas can have multiple part-of-speech tags and thus multiple entries in the lexicon. The path of the lexicon file is `"sent_lexicon"`.)

### 1.1 Naive binary classification
One might look up every word $w_1 ... w_n$ in a document, and compute a **binary score**
$S_{binary}$ by counting how many words have a positive or a
negative label in the sentiment lexicon $SLex$.

[//]: # ($$S_{binary}(w_1 w_2 ... w_n) = \sum_{i = 1}^{n}\text{sign}(SLex\big[w_i\big])$$)

where $\text{sign}(SLex\big[w_i\big])$ refers to the polarity of $w_i$.

**Threshold.** On average, there are more positive than negative words per review (~7.13 more positive than negative per review) to take this bias into account we use a threshold of **8** (roughly the bias itself) to make it harder to classify as positive.

[//]: # ($$
\text{classify}(S_{binary}(w_1 w_2 ... w_n)) = \bigg\{\begin{array}{ll}
        \text{positive} & \text{if } S_{binary}(w_1w_2...w_n) > threshold\\
        \text{negative} & \text{otherwise}
        \end{array}
$$)

This method yields an accuracy of 67.1%.

### 1.2 Binary classfication with magnitude
We take into account the *magnitude* of each word's polarity to calculate the total sentiment:

$$S_{weighted}(w_1w_2...w_n) = \sum_{i = 1}^{n}SLex\big[w_i\big]$$
We achieve an accuracy of 69.3% on the dataset.

### 1.3 Binary classification with relative threshold
We implement a relative threhold based on document length, as longer documents will have greater abolute positive bias than shorter ones. We achieve an accuracy of 69.4% with this approach.

## 2. Naive Bayes
### 2.1 Vanilla Naive Bayes
We implement a simple Naive-Bayes approach that operates on a Bag-of-Words (BoW) representation of the text data.

[//]: # ($$\hat{c} = \operatorname*{arg\,max}_{c \in C} P(c|\bar{f}) = \operatorname*{arg\,max}_{c \in C} P(c)\prod^n_{i=1} P(f_i|c)$$
where $C = \{ \text{POS}, \text{NEG} \}$ is the set of possible classes,
$\hat{c} \in C$ is the most probable class, and $\bar{f}$ is the feature
vector. We use the log of these probabilities when making
a prediction:
$$\hat{c} = \operatorname*{arg\,max}_{c \in C} \Big\{\log P(c) + \sum^n_{i=1} \log P(f_i|c)\Big\}$$)

We use an 9:1 split for train and test sets (with equal positive and negative reviews each) and obtain a test accuracy of 83.5%. 

### 2.2 Smoothing
Repeat the previous experiment but with Laplacian feature smoothing for unseen words.

[//]: # ($$\frac{\text{count}(w_i, c)}{\sum\limits_{w\in V} \text{count}(w, c)}$$ 
for a word
$w_i$ becomes
$$\frac{\text{count}(w_i, c) + \text{smoothing}(w_i)}{\sum\limits_{w\in V} \text{count}(w, c) + \sum\limits_{w \in V} \text{smoothing}(w)}$$
)
Using smoothing results in a test accuracy of 82.5%

### 2.3 Cross-validation
We do 10-fold cross-validation for the NB classifier with round-robin splitting and compare accuracy & variance.

We get an avg accuracy of 81.8% with a variance of 0.006%.

### 2.4 Stemming
We use stemming on the words in the vocabulary to reduce the number of features (dimenionality reduction). The accuracy does not change significantly, but the size of the vocabulary is reduced by ~20%.

### 2.5 N-grams
We retrain the NB classifier from using unigrams+bigrams and unigrams+bigrams+trigrams as features. The accuracy doesn't change significantly, but the size of the vocabulary increases exponentially.

## 3. Support Vector Machines
### 3.1 Linear SVM
We train a SVM classifier using the same train/test split as in the NB classifier. We use a linear kernel and a C value of 1.0. The feature vector is a BoW representation of the text data.
We get an 10-fold cross-validation accuracy of 83.5% on the test set, comparable to the NB classifier.

### 3.2 POS disambiguation
Repeat the experiment with words+POS tags as features. We get an accuracy of 84% over 10-fold cross-validation.

### 3.3 Closed-class words
Repeat the experiment with closed-class words removed. We get an accuracy of 84.3% over 10-fold cross-validation.