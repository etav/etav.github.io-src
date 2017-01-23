---
Title: Animal Shelter Classifier using Random Forest
Slug: random_forest_animal_shelter
Date: 2016-03-15 10:20
Category: Projects
Author: Ernest Tavares III
---

# Animal Shelter
### The Dataset  
Some R code for a kaggle competition I entered. The goal was to create a classifier to predict the outcome of a sheltered animal using features such the animal's gender, age and breed. The training dataset contains 26,729 observations, 9 predictor variables and was given to us by the Austin Animal Shelter.

###Exploratory Analysis
Before I dive into creating a classifier, I typically perform an exploratory analysis moving from observing one univariate statistics to bivariate statistics and finally model building.

However, I broke from my normal process as curiosity got the best of me. I was interested in learning about what the typical outcomes are for sheltered animals (check out the graph below).

![Image of Outcomes]
(https://github.com/etav/animal_shelter/blob/master/img/outcomes_by_animal.png)

Luckily, as we see above, many animals are either adopted, transferred or in the case of dogs frequently returned to their owners.  

###The Variables

```R
[1] "ID"             "Name"           "DateTime"       "OutcomeType"    "OutcomeSubtype" "AnimalType"    
 [7] "SexuponOutcome" "AgeuponOutcome" "Breed"          "Color"    
```

Variable Name | Description
------------ | -------------
ID | The animal's unique ID.
Name | The animal's name, if known (many are not).
DateTime | The date and time the animal entered the shelter (ranges from 1/1/14 - 9/9/15).
OutcomeType | A five factor variable detailing the outcome for the animal (ie: adopted,transferred, died).
OutcomeSubtype | 17 factor variable containing Further details related to the outcome of the animal, such as whether or not they were aggressive.
AnimalType | Whether the animal is a cat or dog.
SexuponOutcome | The sex of the animal at the time the outcome was recorded.
AgeuponOutcome| The age of the animal when the outcome was recorded.
Breed | The breed of the animal (contains mixed breed).
Color | A Description of the coloring on the animal.

###Transforming Variables

The first thing I did was transform the date variable by separating time and date so that I can analyze them independently, I'd like to be able to compare time of day and any seasonality effects on adoption. I then moved on to address missing name values (there were a few mis-codings which caused errors). After that I moved onto transforming the "AgeuponOutcome" variable so that the reported age of animals would all be in the same units, I chose days. This took some chaining of ifelse statements:

####Animal's Age
```R
#Animal Age
split<-str_split_fixed(train$AgeuponOutcome," ", 2) # split value and unit of time
split[,2]<-gsub("s","",split[,2]) #remove tailing "s"

#create a vector to multiply
multiplier <- ifelse(split[,2] == 'day', 1,
                     ifelse(split[,2] == 'week', 7,
                            ifelse(split[,2] == 'month', 30,  
                                   ifelse(split[,2] == 'year', 365, NA))))

train$days_old <- as.numeric(split[,1]) * multiplier #apply the multiplier
train$days_old[1:5] #compare, looks good
train$AgeuponOutcome[1:5]
```

After this transformation, we're able to create a visualization which tells us the outcome of each animal type as a function of its age (in days).

![Image of Age&Outcomes]
(https://github.com/etav/animal_shelter/blob/master/img/age&outcome.png)

Interestingly the likelihood of being adopted for cats varies with age whereas for dogs there appears to be a slight negative correlation between a it's age and the probability it will be adopted.

For dogs, it seems older animals tend to have a higher likelihood of being returned to their owner (I assume this is has to do with the proliferation of chips for animals)

####Animal's Gender
Moving on I decided to compare the differences in outcomes based on the animal's gender. It's clear that adopters favor animals (both cats and dogs) that have previously been neutered. It's interesting to note that a large proportion of cats which were not neutered are transferred to another animal shelter, where (my guess is) they are then neutered.

![Image of Sex&Outcomes]
(https://github.com/etav/animal_shelter/blob/master/img/outcome_by_sex.png)


###Applying Random Forest
After transforming our variables, performing univariate analysis and determining the validity of our sample, it's finally time to move to model building. I will create a random forest using the RandomForest package, using OutcomeType as our predictor variable (remember there are five levels, which complicates things a bit).

```R
rf1 <- randomForest(OutcomeType~AnimalType+SexuponOutcome+Named+days_old+young+color_simple,
                    data=train,importance=T, ntree=500, na.action=na.fail)
rf1
```

![Image of RF_Error]
(https://github.com/etav/animal_shelter/blob/master/img/RF_Error.png)

Our random forest model does poorly at classifying animal deaths which makes sense  when we consider only 197/26729
or 0.007370272% of our training set were flagged as "Died". The model does fairly well at predicting instances where an adoption, transfer, or euthanasia occurs which make up the bulk of the training set. Furthermore, our OOB or out of bag error estimate is  35.28%.

Here's a detailed breakdown of our Random Forest:

```R
#Calling Random Forest
Call:
 randomForest(formula = OutcomeType ~ AnimalType+SexuponOutcome+Named+days_old+young+color_simple, data = train, importance = T,      ntree = 500, na.action = na.fail)
               Type of random forest: classification
                     Number of trees: 500
No. of variables tried at each split: 2

        OOB estimate of  error rate: 35.28%
Confusion matrix:
                Adoption Died Euthanasia Return_to_owner Transfer class.error
Adoption            8988    0          2            1391      388   0.1653821
Died                  21    0         10              10      156   1.0000000
Euthanasia           229    0        175             380      771   0.8874598
Return_to_owner     1998    0          9            2393      386   0.5000000
Transfer            2641    0         83             954     5744   0.3903630
```

###Determining Variable Importance
Finally, we'll rank our predictor variables based on their mean reduction in Gini error.

![Image of var_importance]
(https://github.com/etav/animal_shelter/blob/master/img/var_importance.png)


As we see in the graphic above, the animal's sex and age were most useful to reduce the mean Gini Error. Interestingly, animal type (ie: cat or dog) and the physical features of the animal such as the color mattered less.

That's all for this one folks, thanks for tuning in!
