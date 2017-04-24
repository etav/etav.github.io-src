---
Title: Implementing Bayes' Theorem from Scratch
Slug: bayes_theorem_scratch
Date: 2017-04-23 5:20
Category: Algorithms
Author: Ernest Tavares III
---

## Implementing Bayes Theorem from Scratch

Baye's theorem calculates the probability of a given class or state given the joint-probability distribution of the input variables (betas). There are numerous libraries which take care of this for us which are native to python and R but in order to understand what's happening "behind the scenes" we'll use bayes theorem to calculate join probability distributions from scratch.

Here we'll create a ficticuous world in which we're a doctor testing patients for HIV, given these assumptions:

**`P(HIV)`** = The odds of a person having HIV is .015 or 1.5%

**`P(Positive)`** = The probability the test results are positive

**`P(Negative)`** = The probability the test results are negative.

**`P(Positive | HIV)`** = The probability the test results are positive given someone has HIV. This is also called Sensitivity or True Positive Rate. We'll assume the test is correct .95 or 95% of the time.

**`P(Positive | ~HIV)`** = The probability the test results are positive given someone does not have HIV. This is also called Specificity or True Negative Rate. We'll assume this is also correct .95 or 95% of the time.

Baye's Formula:

![img](http://www.idgconnect.com/IMG/313/9313/formula-image.jpg)

Where:
- `P(A)` is the prior probability of A occuring independently, for us this is `P(HIV)`.
- `P(B)` is the prior probability of B occuring independently, for us this is `P(Positive)`.
- `P(A|B)` is the posterior probability of A occuring given B occurs, for us this is `P(HIV | Positive)`. This is the probability that an individual has HIV given their test results are positive and what we're trying to calculate.
- `P(B|A)` is the likelihood probability of B occuring, given A occus. In our example this is `P(Positive | HIV)`. This value is given to us.

Strining these together we get:

`P(HIV | Positive) = ((P(HIV) * P(Positive | HIV)) / P(Positive)`

Thus the probability of getting a positive HIV test result `P(HIV)` becomes:

`P(Positive) = [P(HIV) * Sensitivity] + [P(~HIV) * (1-Specificity)]`

## Calculations - Priors


```python
#perfoming calculations:

p_hiv = .015 #P(HIV) assuming 1.5% of the population has HIV

p_no_hiv = .98 # P(~HIV)

p_positive_hiv = .95 #sensitivity

p_negative_hiv = .95#specificity

#P(Positive)
p_positive = (p_hiv * p_positive_hiv) + (p_no_hiv * (1-p_negative_hiv))
print "The probability of getting a positive test result is:", p_positive, "this is our prior"
```

    The probability of getting a positive test result is: 0.06325 this is our prior


Using this prior we can calculate our posterior probalities as follows:

The probability of an individual having HIV given their test result is positive.

`P(D|Positive) = (P(HIV) * Sensitivity)) / P(Positive)`

The probability of an individual not having HIV given their test result is positive.

`P(~D|Positive) = (P(~HIV) * (1-Sensitivity))) / P(Positive)`

**Note: the sum of posteriors must equal one because combined they capture all possible states within our set of probabilities.**

## Calculations - Posteriors


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

print "The probability of an individual not having HIV a positive test result is:", p_no_hiv_positive
```

    The probability of an individual not having HIV a positive test result is: 0.774703557312


## Conclusion

That's it! We've just demonstrated how to calculate Bayes theorem from scrath. In our toy example we showed that if an individual gets a positive test result the probability this individual has HIV is 22.5% and 77.5% that they do not have HIV. We can check the validity of our results by summing the probability of both cases:


```python
posterior_sum = p_no_hiv_positive + p_hiv_positive
posterior_sum #sum to 1, looks good!
```




    1.0
