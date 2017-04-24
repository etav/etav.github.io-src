---
Title: Spam Classifier using Naive Bayes
Slug: spam_message_classifier_naive_bayes
Date: 2017-04-23 10:20
Category: Projects
Author: Ernest Tavares III
---

## 1 Import Dataset

The SMS spam dataset can be downloaded [here](https://archive.ics.uci.edu/ml/machine-learning-databases/00228/).


```python
import pandas as pd
# Dataset from - https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection
df = pd.read_table('smsspamcollection/SMSSpamCollection',
                   sep='\t',
                   header=None,
                   names=['label', 'sms_message'])

df.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>label</th>
      <th>sms_message</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ham</td>
      <td>Go until jurong point, crazy.. Available only ...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ham</td>
      <td>Ok lar... Joking wif u oni...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>spam</td>
      <td>Free entry in 2 a wkly comp to win FA Cup fina...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ham</td>
      <td>U dun say so early hor... U c already then say...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ham</td>
      <td>Nah I don't think he goes to usf, he lives aro...</td>
    </tr>
  </tbody>
</table>
</div>



## 1.1 Process the data set
We need to transform the labels to binary values so we can run the regression. Here 1 = "spam" and 0 = "ham"


```python
#Map applies a function to all the items in an input list or df column.
df['label'] = df.label.map({'ham':0, 'spam':1})
df.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>label</th>
      <th>sms_message</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>Go until jurong point, crazy.. Available only ...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>Ok lar... Joking wif u oni...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>Free entry in 2 a wkly comp to win FA Cup fina...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>U dun say so early hor... U c already then say...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>Nah I don't think he goes to usf, he lives aro...</td>
    </tr>
  </tbody>
</table>
</div>



## 2.1 Enter Bag of Words
Since we're dealing with text data and the naive bayes classifier is better suited to having numerical data as inputs we will need to perform transformations. To accomplish this we'll use the ("bag of words")[https://en.wikipedia.org/wiki/Bag-of-words_model] method to count the frequency of occurance for each word. Note: the bag of word methods assumes equal weight for all words in our "bag" and does not consider the order of occurance for words.

There are modules that will do this for us but we will implement bag of words from scratch to understand what's happening under the hood.

The steps are as follow:
1. Convert bag of words to lowercase.
2. Remove punctuation from sentences.
3. Break on each word.
4. Count the frequency of each word.



```python
import string #punctuation
import pprint
from collections import Counter #frequencies

#Bag of Words from scratch
documents = ['Hello, how are you!',
             'Win money, win from home.',
             'Call me now.',
             'Hello, Call hello you tomorrow?']

lower_case_documents = []

for i in documents:
    lower_case_documents.append(i.lower())
print "lower case:", lower_case_documents

# Remove punctuation.
sans_punctuation_documents = []

for i in lower_case_documents:
    sans_punctuation_documents = ["".join( j for j in i if j not in string.punctuation) for i in  lower_case_documents]
print"no punctuation:", (sans_punctuation_documents)

#Break each word
preprocessed_documents = []

for i in sans_punctuation_documents:
    preprocessed_documents.append(i.split(' ')) #split on space
print "break words:", (preprocessed_documents)

#Count frequency of words using counter
frequency_list = []

for i in preprocessed_documents:
    frequency_counts = Counter(i)
    frequency_list.append(frequency_counts)
print "tokenized counts:", pprint.pprint(frequency_list)
```

    lower case: ['hello, how are you!', 'win money, win from home.', 'call me now.', 'hello, call hello you tomorrow?']
    no punctuation: ['hello how are you', 'win money win from home', 'call me now', 'hello call hello you tomorrow']
    break words: [['hello', 'how', 'are', 'you'], ['win', 'money', 'win', 'from', 'home'], ['call', 'me', 'now'], ['hello', 'call', 'hello', 'you', 'tomorrow']]
    tokenized counts:[Counter({'how': 1, 'you': 1, 'hello': 1, 'are': 1}),
     Counter({'win': 2, 'home': 1, 'from': 1, 'money': 1}),
     Counter({'me': 1, 'now': 1, 'call': 1}),
     Counter({'hello': 2, 'you': 1, 'call': 1, 'tomorrow': 1})]
     None


## 2.2 SciKit-Learn Feature Extraction
That was pretty simple but scikit-learn makes the process even easier. Let's try it using the `sklearn.feature_extraction.text.CountVectorizer` method from the module.  


```python
from sklearn.feature_extraction.text import CountVectorizer
count_vector = CountVectorizer() #set the variable

count_vector.fit(documents) #fit the function
count_vector.get_feature_names() #get the outputs
```




    [u'are',
     u'call',
     u'from',
     u'hello',
     u'home',
     u'how',
     u'me',
     u'money',
     u'now',
     u'tomorrow',
     u'win',
     u'you']



Create an array where each row represents one of the 4 columns and each column represents the counts for each word within the document.


```python
doc_array = count_vector.transform(documents).toarray()
doc_array
```




    array([[1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1],
           [0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 2, 0],
           [0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0],
           [0, 1, 0, 2, 0, 0, 0, 0, 0, 1, 0, 1]])



Convert the array to a data frame and apply get_feature_names as the column names.


```python
frequency_matrix = pd.DataFrame(doc_array,
                                columns = count_vector.get_feature_names()
                               )
frequency_matrix
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>are</th>
      <th>call</th>
      <th>from</th>
      <th>hello</th>
      <th>home</th>
      <th>how</th>
      <th>me</th>
      <th>money</th>
      <th>now</th>
      <th>tomorrow</th>
      <th>win</th>
      <th>you</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



## 3.1 Training & Testing Sets

We'll split our dataset using scikit's `train_test_split` method into training and testing sets so we can make inferences about the model's accuracy on data it hasn't been trained on.


```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df['sms_message'],
                                                    df['label'],
                                                    random_state=1)

print "Our original set contains", df.shape[0], "observations"
print "Our training set contains", X_train.shape[0], "observations"
print "Our testing set contains", X_test.shape[0], "observations"
```

    Our original set contains 5572 observations
    Our training set contains 4179 observations
    Our testing set contains 1393 observations


Fit the training & testing data to the `CountVectorizer()` method and return a matrix


```python
train = count_vector.fit_transform(X_train)
test = count_vector.transform(X_test)
```

## 4.1 Implementing Baye's Theorem from Scratch

Bayes' theorem calculates the probability of a given class or state given the joint-probability distribution of the input variables (betas). There are numerous libraries which take care of this for us native to python and R but in order to understand what's happening behind the scenes let's calculate bayes theorem from scratch.

Here we'll create a fictitious world in which we're testing patients for HIV.

**P(HIV)** = The odds of a person having HIV is .015 or 1.5%

**P(Positive)** = The probability the test results are positive

**P(Negative)** = The probability the test results are negative.

**P(Positive | HIV)** = The probability the test results are positive given someone has HIV. This is also called Sensitivity or True Positive Rate. We'll assume the test is correct .95 or 95% of the time.

**P(Positive | ~HIV)** = The probability the test results are positive given someone does not have HIV. This is also called Specificity or True Negative Rate. We'll assume this is also correct .95 or 95% of the time.

Baye's Formula:

![img](http://www.idgconnect.com/IMG/313/9313/formula-image.jpg)

Where:
- `P(A)` is the probability of A occurring independently, for us this is `P(HIV)`.
- `P(B)` is the probability of B occurring independently, for us this is `P(Positive)`.
- `P(A|B)` is the posterior probability of A occurring given B occurs, for us this is `P(HIV | Positive)`. This is the probability that an individual has HIV given their test results are positive and what we're trying to calculate.
- `P(B|A)` is the likelihood probability of B occurring, given A occurs. In our example this is `P(Positive | HIV)`. This value is given to us.

Stringing these together we get:

`P(HIV | Positive) = ((P(HIV) * P(Positive | HIV)) / P(Positive)`

Thus the probability of getting a positive HIV test result `P(HIV)` becomes:

`P(Positive) = [P(HIV) * Sensitivity] + [P(~HIV) * (1-Specificity)]`


```python
#performing calculations:


p_hiv = .015 #P(HIV) assuming 1.5% of the population has HIV

p_no_hiv = .98 # P(~HIV)

p_positive_hiv = .95 #sensitivity

p_negative_hiv = .95#specificity

#P(Positive)
p_positive = (p_hiv * p_positive_hiv) + (p_no_hiv * (1-p_negative_hiv))
print "The probability of getting a positive test result is:", p_positive, "this is our prior"
```

    The probability of getting a positive test result is: 0.06325 this is our prior


Using this prior we can calculate our posterior probabilities as follows:

The probability of an individual having HIV given their test result is positive.

`P(D|Positive) = (P(HIV) * Sensitivity)) / P(Positive)`

The probability of an individual not having HIV given their test result is positive.

`P(~D|Positive) = (P(~HIV) * (1-Sensitivity))) / P(Positive)`

**Note: the sum of posteriors must equal one because combined they capture all possible states within our set of probabilities.**


```python
#P(HIV | Positive)
p_hiv_positive = (p_hiv * p_positive_hiv) / p_positive

print "The probability of a person having HIV, given a positive test result is:", p_hiv_positive
```

    The probability of a person having HIV, given a positive test result is: 0.225296442688



```python
#P(~HIV | Positive)
p_positive_no_hiv = 1 - p_positive_hiv
p_no_hiv_positive = (p_no_hiv * p_positive_no_hiv) / p_positive

print "The probability of an individual not having HIV given getting a positive test result is:", p_no_hiv_positive
```

    The probability of an individual not having HIV given getting a positive test result is: 0.774703557312


That's it! We've just demonstrated how to calculate Bayes theorem from scratch. In our toy example we showed that if an individual gets a positive test result the probability this individual has HIV is 22.5% and 77.5% that they do not have HIV. We can check the validity of our results by summing the probability of both cases:


```python
posterior_sum = p_no_hiv_positive + p_hiv_positive
posterior_sum #sum to 1, looks good!
```




    1.0



## 5.1 Naive Bayes Classifier using Scikit-Learn

In the above example we only the probability given two inputs (the test result and the status of the disease in the patient). This calculation would grow exponentially more complex given numerous inputs and would be painstaking to calculate by hand. Don't worry, SciKit-Learn is here to save the day (and a ton of time)!

Our spam classifier will use multinomial naive Bayes method from `sklearn.nive_bayes`. This method is well-suited for for discrete inputs (like word counts) whereas the Gaussian Naive Bayes classifier performs better on continuous inputs.


```python
from sklearn.naive_bayes import MultinomialNB
naive_bayes = MultinomialNB() #call the method
naive_bayes.fit(train, y_train) #train the classifier on the training set
```




    MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)




```python
predictions = naive_bayes.predict(test) #predic using the model on the testing set
```

## 6.1 Evaluating the Model

After training our model we're now ready to evaluate its accuracy and precision.

- `Accuracy:` A ratio of correct predictions to the total number of predictions.
- `Precision:` The proportion of messages which were correctly classified as spam. This is a ratio of true positives (messages classified as SPAM which actually are SPAM) to all positives (all messages classified as SPAM).


```python
from sklearn.metrics import accuracy_score, precision_score,f1_score

print('accuracy score: '),format(accuracy_score(y_test,predictions))
print('precision score: '),format(precision_score(y_test,predictions))
```

    accuracy score:  0.988513998564
    precision score:  0.972067039106


## Conclusion
Through this excercise we learned how to implement bag of words and the naive bayes method first from scratch to gain insight into the technicalities of the methods and then again using scikit-learn to provide scalable results.

We've learned that the naive bayes classifier can produce robust results without significant tuning to the model.

Our final model classifies text messages as spam with 98.8% accuracy and 98.8% precision.
