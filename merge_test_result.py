
test_file = 'models/edc-mbert-cased-combined/test_predictions.txt'

prediction_file = 'raw_predictions.txt'

pred_file = open(prediction_file, 'r')

preds = []
for line in pred_file:
    line = line.strip()
    if line:
        preds.append(line)

pred_file.close()

i = 0

out_file = open('test_results.txt', 'w')
test_file = open(test_file, 'r')

for line in test_file:
    line = line.strip()
    if line:
        word, tag = line.split()
        if tag != 'O':
            out_file.write('%s %s\n' % (word, preds[i]))
            i += 1
        else:
            out_file.write('%s %s\n' % (word, tag))
    else:
        out_file.write('\n')

test_file.close()
out_file.close()