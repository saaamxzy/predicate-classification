

if __name__ == '__main__':

    detection_output = 'data/event-detection-classification/dev.txt'
    classification_output = 'models/.txt'

    classification_prediction_file = open(classification_output, 'r')

    classification_preds = []
    for line in classification_prediction_file:
        line = line.strip()
        if line:
            classification_preds.append(line)

    classification_prediction_file.close()

    i = 0

    out_file = open('combined_predictions.txt', 'w')
    detection_prediction_file = open(detection_output, 'r')

    for line in detection_prediction_file:
        line = line.strip()
        if line:
            word, tag = line.split()
            if tag != 'O':
                out_file.write('%s %s\n' % (word, classification_preds[i]))
                i += 1
            else:
                out_file.write('%s %s\n' % (word, tag))
        else:
            out_file.write('\n')

    detection_prediction_file.close()
    out_file.close()
