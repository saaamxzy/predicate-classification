

source_file_path = 'ontonotes_arabic_better.txt'
dest_file_path = 'extra_training.txt'

source_file = open(source_file_path, 'r')

num_sentences = 0
sentence = []
predicate_idx = []

words_ls = []
predicate_idx_ls = []

for line in source_file:
    line = line.strip()
    words = line.split('|||')[0].strip().split()
    idx = words[0]
    words = words[1:]
    tags = line.split('|||')[1].strip().split()
    if words == sentence:
        for i,tag in enumerate(tags):
            if tag == 'B-V':
                predicate_idx[i] = 'B'
            elif tag == 'I-V':
                predicate_idx[i] = 'I'
    else:
        if sentence:
            words_ls.append(sentence)
            predicate_idx_ls.append(predicate_idx)

        sentence = words
        predicate_idx = ['O' for _ in range(len(sentence))]


source_file.close()


dest_file = open(dest_file_path, 'w')

assert len(words_ls) == len(predicate_idx_ls)

for i in range(len(words_ls)):
    words = words_ls[i]
    predicate_idx = predicate_idx_ls[i]
    assert len(words) == len(predicate_idx)
    for j in range(len(words)):
        dest_file.write('%s %s\n' % (words[j], predicate_idx[j]))
    dest_file.write('\n')


dest_file.close()