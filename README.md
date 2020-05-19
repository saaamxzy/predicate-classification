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
|Precision|84.0| |47.6|
|Recall|88.0| |5.9|
|F1|85.9|   |10.5|


### Adding Extra Training Data

|Metric|English Dev|English Test|Arabic Test|
|---|---|---|---|
|Precision|84.6| |86.3|
|Recall|84.4| |81.8|
|F1|84.5|   |84.0|

## Predicate Classification

### Predicate as Segment B

### Adding Predicate Position Indicator

### Adding Max and Mean pooling over BERT embeddings

## Combining the Two Models