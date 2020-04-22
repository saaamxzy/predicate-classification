import os


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None, predicate_vector=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
            predicate: (Optional) string. The binary vector indicating the location
            of the predicate word in a sentence.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.predicate_vector = predicate_vector

    def __str__(self):
        return 'guid: %s\ttext_a: %s\ttext_b: %s\tlabel: %s\tpredicate: %s' % (self.guid, self.text_a, self.text_b,
                                                                               self.label, self.predicate_vector)


def get_labels_from_file(path):
    if path:
        with open(path, "r") as f:
            labels = f.read().splitlines()
        return labels
    else:
        return []


def read_examples_from_file(data_dir, mode):
    file_path = os.path.join(data_dir, "{}.txt".format(mode))
    guid_index = 1
    examples = []

    label_dict = {}

    with open(file_path, encoding="utf-8") as f:
        words = []
        # labels = []
        label = None
        predicate = None
        predicate_vector = []
        num_sentences = 0
        for line in f:
            line = line.strip()
            if line == "":
                if words:
                    text_a = " ".join(words)
                    examples.append(InputExample(guid="{}-{}".format(mode, guid_index), text_a=text_a, text_b=predicate,
                                                 label=label, predicate_vector=predicate_vector))
                    guid_index += 1
                    words = []
                    num_sentences += 1
                    predicate_vector = []
            else:
                splits = line.split(" ")
                words.append(splits[0])
                if splits[1] != 'O':
                    label = splits[1]
                    predicate = splits[0]
                    predicate_vector.append(1)
                    if label not in label_dict:
                        label_dict[label] = 1
                    else:
                        label_dict[label] += 1
                else:
                    predicate_vector.append(0)

    print('Total number of examples loaded: ', num_sentences)
    return examples, label_dict


def print_dict(label_dict):
    total = sum(label_dict.values())

    for label, count in sorted(label_dict.items(), key=lambda x:x[0]):
        print('%s : %d, percentage: %f' % (label, count, count / float(total)))


if __name__ == '__main__':
    labels = get_labels_from_file('./data/predicate-classification/labels.txt')

    train_examples, train_label_dict = read_examples_from_file('./data/predicate-classification', 'train')
    print(len(train_examples))
    print('train:')
    print_dict(train_label_dict)

    examples, label_dict = read_examples_from_file('./data/predicate-classification', 'dev')
    print(len(examples))
    print('dev:')
    print_dict(label_dict)