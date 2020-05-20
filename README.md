# Predicate Detection and Classification

This project uses BERT to construct models that can be used in event trigger 
detection and classification. The BERT model used in these experiments is the
bert-base-multilingual-cased version. The task is modeled as a pipeline of two 
tasks: the detector and classifier.

## Predicate Detection
The first task in this pipeline. The detector model takes in a sequence of words
and outputs a sequence tag for each words in the BIO format.

### Detector Training
A predicate detector is trained using the command provided in run_detect.sh,
where information about hyper-parameter choices could also be found.
Note the number of epochs is small because we are only fine-tuning the BERT
model on this sequence tagging task. The model performance is given in the
following table:

|Metric|English Dev|English Test|Arabic Test|
|---|---|---|---|
|Precision|84.0|85.2|47.6|
|Recall|88.0|86.9|5.9|
|F1|85.9|86.0|10.5|


### Adding Extra Training Data

In the training process of the last model, there is no arabic training data
hence the performance on arabic data is not good. We add extra arabic and english
training data to the training set and trained the same model with same 
hyper-parameters. The results are shown below:

|Metric|English Dev|English Test|Arabic Test|
|---|---|---|---|
|Precision|84.6|86.9|86.3|
|Recall|84.4|83.7|81.8|
|F1|84.5|85.3|84.0|

More english training data did not increase the scores on english dev and test
sets. However, the arabic training data did boost the performance on the arabic
test set. 

## Predicate Classification
In this task, we trained models to classify a predicate in a sentence into
one of the 12 classes. If a sentence contains N predicates, we split
it into N data points, each corresponding to a predicate of the sentence.
Each data point consists of an input and a label, where the input is the sentence plus the
location of the predicate word, and the label is one of the 12 classes.


### Model A: Predicate as Segment B
In this model we did not use the location indicator of the predicate. The 
architecture is similar to the one below:

Model performance (based on BIO format sequence using 
[seqeval](https://github.com/chakki-works/seqeval)):

|Metric|English Dev|English Test|Arabic Test|
|---|---|---|---|
|Precision|59.8|61.7|36.2|
|Recall|64.9|66.5|47.2|
|F1|62.2|64.0|41.0|

### Adding Predicate Position Indicator
In this model we use the location indicator of the predicate. The architecture
of this model is as follow:

Model performance:

|Metric|English Dev|English Test|Arabic Test|
|---|---|---|---|
|Precision|60.5|59.8|36.2|
|Recall|65.5|64.5|47.2|
|F1|62.9|62.0|41.0|


### Adding Max and Mean pooling over BERT embeddings

Model performance:

|Metric|English Dev|English Test|Arabic Test|
|---|---|---|---|
|Precision|60.5|59.8|36.2|
|Recall|65.5|64.5|47.2|
|F1|62.9|62.0|41.0|

####12-class classification accuracy for all three models:

|Model|English Dev|English Test|Arabic Test|
|---|---|---|---|
|A|69.1|69.1|55.3|
|B|69.6|xx.x|55.3|
|C|xx.x|xx.x|xx.x|

## A Different Approach

We also tried further break down the classification task into two classification
tasks: one with three classes and another with four classes. The final 
classification result is given by combining the prediction results of the two
sub-tasks. The first classifier classifies a sentence and a given predicate into 
one of **neutral**, **harmful** and **helpful** whereas the second classifier
classifies the same input into one of **verbal**, **material**, **both** and 
**unknown**. However, we did not observe any increase in accuracy score as the
final combined accuracy was only 0.71.


## Running the Code