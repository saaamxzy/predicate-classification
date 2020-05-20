import os

edc_dir = 'event-detection-classification/'

edc_train_path = edc_dir + 'train.txt'
edc_dev_path = edc_dir + 'dev.txt'
edc_test_path = edc_dir + 'test.txt'
edc_labels_path = edc_dir + 'labels.txt'

pc_dir = 'predicate-classification/'

pc_train_path = pc_dir + 'train.txt'
pc_dev_path = pc_dir + 'dev.txt'
pc_test_path = pc_dir + 'test.txt'
pc_labels_path = pc_dir + 'labels.txt'


def write_pc_data(mode: str):
    print('processing ' + mode + ' data...')
    edc_file = edc_train_path
    pc_file = pc_train_path
    no_label = False
    if mode == 'train':
        edc_file = edc_train_path
        pc_file = pc_train_path
    elif mode == 'dev':
        edc_file = edc_dev_path
        pc_file = pc_dev_path
    elif mode == 'test':
        edc_file = edc_test_path
        pc_file = pc_test_path
    elif mode == 'arab':
        edc_file = edc_dir + 'arabic/test.txt'
        pc_file = pc_dir + 'test_arab.txt'
    else:
        edc_file = 'models/edc-mbert-cased-combined/test_predictions.txt'
        pc_file = 'data/test_predictions.txt'
        no_label = True

    in_file = open(edc_file, 'r')

    examples = []
    labels = []
    words = []
    tags = []

    counter = 0
    for line in in_file:
        line = line.strip()
        if not line:
            # now words variable has words of a sentence and tags variable has tags for all words in the sentence
            assert len(words) == len(tags)  # else there is missing tag or word.
            for i in range(len(words)):
                w = words[i]
                t = tags[i]
                new_tags = ['O' for _ in range(len(tags))]
                if t != 'O':
                    # we found a predicate, need to create a stand-alone data point for it
                    if not no_label:
                        t = t[2:]  # remove the 'B-' or 'I-'
                    new_tags[i] = t
                    examples.append(words)
                    labels.append(new_tags)
            counter += 1
            words = []
            tags = []
        else:
            line = line.split()
            word, tag = line[0], line[1]
            words.append(word)
            tags.append(tag)

    in_file.close()

    print('Total number of examples in original file: %d' % counter)
    print('Total number of examples in new pc file: %d' % len(examples))

    out_file = open(pc_file, 'w')
    for i in range(len(examples)):
        example = examples[i]
        label = labels[i]
        n = len(example)
        for j in range(n):
            out_file.write(example[j] + ' ' + label[j] + '\n')
        out_file.write('\n')

    out_file.close()


def main():
    # write_pc_data('train')
    # write_pc_data('dev')
    # write_pc_data('test')
    write_pc_data('arab')
    # write_pc_data('else')


if __name__ == '__main__':
    if not os.path.isdir(pc_labels_path):
        main()
    else:
        print('Data exists. Overwriting existing data...')
        main()

