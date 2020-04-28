

source_file_path = 'ontonotes_arabic_better.txt'
dest_file_path = 'extra_training.txt'

source_file = open(source_file_path, 'r')

num_sentences = 0
sentence = ''

for line in source_file:
    line = line.strip()
    words = line.split('|||')[0].strip().split()
    idx = words[0]
    words = words[1:]
    tags = line.split('|||')[1].strip().split()



source_file.close()


dest_file = open(dest_file_path, 'w')


dest_file.close()