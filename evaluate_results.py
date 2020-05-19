from seqeval import metrics

true_label = []
words = []
pred_label = []

true_label_set = set()
pred_label_set = set()

true_file = open('data/event-detection-classification/test.txt', 'r')

sentence = []
label = []

for line in true_file:
    line = line.strip()
    if line:
        word, tag = line.split()

        if tag != 'O':
            true_label_set.add(tag[2:])
            tag = tag.replace('---', '+')
        else:
            true_label_set.add(tag)
        sentence.append(word)
        label.append(tag)

    else:
        true_label.append(label)
        label = []

true_file.close()

print(len(true_label))

pred_file = open('test_results.txt', 'r')

for line in pred_file:
    line = line.strip()
    if line:
        word, tag = line.split()
        pred_label_set.add(tag)
        if tag != 'O':
            tag = tag.replace('---', '+')
            if not label or label[-1] != tag:
                tag = 'B-' + tag
            else:
                # label and label[-1] == tag
                tag = 'I-' + tag
        sentence.append(word)
        label.append(tag)

    else:
        pred_label.append(label)
        label = []

pred_file.close()

assert len(true_label) == len(pred_label)

print('f1: %f' % (metrics.f1_score(true_label, pred_label)))
print('precision: %f' % (metrics.precision_score(true_label, pred_label)))
print('recall: %f' % (metrics.recall_score(true_label, pred_label)))

print('acc: %f' % (metrics.accuracy_score(true_label, pred_label)))

print(metrics.classification_report(true_label, pred_label))

print('true set: ', true_label_set)

print('pred set: ', pred_label_set)