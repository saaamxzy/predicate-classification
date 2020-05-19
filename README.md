# Predicate Detection and Classification

This project uses BERT to construct models that can be used in event trigger 
detection and classification. The BERT model used in these experiments is the
bert-base-multilingual-cased version. The task is modeled as a pipeline of two 
tasks: the detector and classifier.

## Predicate Detection
The first task in this pipeline. The detector model takes in a sequence of words
and outputs a sequence tag for each words in the BIO format.

### Detector Training
A predicate detector is trained using the command provided in run_detect.sh.
Note the number of epochs is small because we are only fine-tuning the BERT
model on this sequence tagging task. The model performance is given in the
following table:

|   |English Dev|English Test|Arabic Test|
|---|---|---|---|
|   |   |   |   |
|   |   |   |   |
|   |   |   |   |

### Adding Extra Training Data


## Predicate Classification

### Predicate as Segment B

### Adding Predicate Position Indicator

### Adding Max and Mean pooling over BERT embeddings

## Combining the Two Models