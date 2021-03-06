# Predicate Detection and Classification

This project uses BERT to construct models that can be used in event trigger 
detection and classification. The task is modeled as a pipeline of two 
tasks: the detector and classifier. We use uncased mBERT with the detection 
task and cased mBERT for the classification task. 

# Highlight

Using the extra training data from the SRL task, we improved the detector
scores on arabic data. As a result, the final classification score would
also increase on arabic data. Note all performance score in this section
are in F1 score.

Detector performance:

|Model|English Dev|English Test|Arabic Test|
|---|---|---|---|
|Without Extra Training|85.9|86.0|36.9|
|With Extra Training|84.5|85.3|**84.0**|

Original detector model without extra training data has a F1 score of 10.5 on
the arabic test data we use, while the model trained with extra English and 
arabic data has an F1 score of **84.0**.

Detector + Classification BIO F1 score:

|Model|English Dev|English Test|
|---|---|---|
|Without Extra Training|57.1|57.1|
|With Extra Training|56.7|57.0|

The scores are decreased on the English data because of the extra training
data on the detector. However, as the detector scored much higher with the
 extra arabic training data it is likely that the final result for additional
 arabic data will be improved. We did not evaluate on the original arabic
 test data because the size of the dataset is too small(10 sentences) and not representative.
 We did not have 12-class labels for the extra arabic data
so we could only show the arabic classification scores once we obtain more
labeled data on the arabic set. The classifier could not perform as well as
the detector. Please see the Predicate Classification and Tuning the Classification
Model sections for possible reasons why the classifier was not doing good.

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
|Precision|84.0|85.2|54.5|
|Recall|88.0|86.9|27.9|
|F1|85.9|86.0|36.9|


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

### Note
We found the original arabic test dataset to be very small(~10 sentences) so we
build and test with a subset(~20%) of arabic data from the extra training data
we obtained from SRL dataset. We still give the performance of our model on the
original arabic test data here just as a reference:

|Metric|Without extra training|With extra training|
|---|---|---|
|Precision|54.5|63.2|
|Recall|27.9|27.9|
|F1|36.9|38.7|

The scores only increased a little. We think the original arabic test set is
too small and thus not representative.

## Predicate Classification
In this task, we trained models to classify a predicate in a sentence into
one of the 12 classes. If a sentence contains N predicates, we split
it into N data points, each corresponding to a predicate of the sentence.
Each data point consists of an input and a label, where the input is the sentence plus the
location of the predicate word, and the label is one of the 12 classes. Note
that in tasks involving classification we will use the original arabic test
data because there are no labels available for the extra arabic test data
acquired from the SRL training data.


### Predicate as Segment B
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
|Precision|61.5|60.4|38.3|
|Recall|66.6|65.2|50.0|
|F1|64.0|62.7|43.4|

### Combined Performance on English (Detection + Classification)

See highlight section.

## Tuning the Classification Model

We observed that the detector had a fair performance but the classifier is
performing poorly. So we experimented in several ways to see if we could
boost its scores.

The model had a training score of nearly 100, which could be an indicator of
overfitting. We tried a number of approaches to decrease the influence of 
overfitting, including decreasing the hidden size of the MLP at the end, or
even removing the hidden layer of the MLP. We also tried adding a dropout
layer to the MLP, but none of those helped the classification scores. Then
we tried to decrease the hidden size of the BiLSTM layer, from 768 to 384,
and finally to 32, 16 or 8. We did not observe any improvement. We also
built another model, who has a Transformer layer with attention on top of
BERT, and it did not outperform the original one. Also, freezing the BERT
layers during training process did not help.

We also tried further break down the classification task into two classification
tasks: one with three classes and another with four classes. The final 
classification result is given by combining the prediction results of the two
sub-tasks. The first classifier classifies a sentence and a given predicate into 
one of **neutral**, **harmful** and **helpful** whereas the second classifier
classifies the same input into one of **verbal**, **material**, **both** and 
**unknown**. However, we did not observe any increase in accuracy score as the
final combined accuracy and F1-score are no higher than the original model.

# How to run the code

## Detector

To train a detector model, run run_detector.sh file.

To get the prediction file from detector, run run_detector_eval.sh. Note the
TEST_FILE macro refers to the file you want to generate predictions under
the data_dir. The file name should be in .txt format and set the macro to
the name of the file without the .txt extension.

## Classifier

To train a classifier model, run the run_classifier.sh file.

To get the prediction file from classifier, run run_classifier_pred.sh.
The detector_prediction_file is the path to the detector output. output_dir
is the path to the directory where the trained model is stored.

## Evaluate the final result with Seqeval

Run the evaluate.sh file. The two arguments refer to the true answer file
and the prediction file in the same format.

## Pretrained models

You can download the pretrained detector model and classifier model
[here](https://drive.google.com/file/d/1YrrqTaPjMu3v_wKxL9LqJlrt9ghcShFz/view?usp=sharing).
Down load and unzip them then change the path to these models in the scripts.
