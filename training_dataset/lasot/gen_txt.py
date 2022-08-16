import os

seq_path = 'data'
seq_name = os.listdir(seq_path)
with open('test_id.txt', 'w') as f:
    for seq in seq_name:
        f.write(seq+'\n')