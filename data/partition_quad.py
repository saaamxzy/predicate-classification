import os

mode = 'dev'
quad = 'first_quad'

source_file_path = 'predicate-classification/'

first_target_file_path = 'first_quad/'
second_target_file_path = 'second_quad/'

for mode in ['train', 'dev', 'test']:

    in_file_path = source_file_path+mode+'.txt'
    first_out_file_path = first_target_file_path+mode+'.txt'
    second_out_file_path = second_target_file_path+mode+'.txt'

    in_file = open(in_file_path, 'r')
    first_out_file = open(first_out_file_path, 'w')
    second_out_file = open(second_out_file_path, 'w')

    for line in in_file:
        if '---' in line:
            line = line.strip()
            word, label = line.split(' ')
            label_1, label_2 = label.split('---')
            first_out_file.write(word+' ' + label_1+'\n')
            second_out_file.write(word+' ' + label_2 + '\n')
        else:
            first_out_file.write(line)
            second_out_file.write(line)

    in_file.close()
    first_out_file.close()
    second_out_file.close()
