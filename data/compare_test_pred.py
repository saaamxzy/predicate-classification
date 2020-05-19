pred_file = open('test_predictions.txt', 'r')
preds = pred_file.readlines()
pred_file.close()

true_file = open('test.txt', 'r')
trues = true_file.readlines()
true_file.close()

assert len(preds) == len(trues)

new_pred_file = open('test_predictions_adjusted.txt', 'w')

for i in range(len(preds)):
    pred = preds[i]
    true = trues[i]
    if pred != true:
        if true.strip().split()[1] == 'O':
            # false positive for a predicate, keeping the true label
            new_pred_file.write(true)
        else:
            # false negative for a predicate, keeping the pred label
            new_pred_file.write(pred)

new_pred_file.close()
