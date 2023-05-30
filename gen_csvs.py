import csv
import sys
import tqdm

nq_file = sys.argv[1]
t0_file = sys.argv[2]
prefix = sys.argv[3]

nq_file = csv.reader(open(nq_file), delimiter='\t')
t0_file = open(t0_file).readlines()
idx = 0
for nq in tqdm.tqdm(nq_file):
    question, answers = nq
    question = question.strip()
    if question[-1] == '?':
        question = question[:-1]
    fw = open(prefix+"-%d.csv"%idx, 'w')
    cw = csv.writer(fw, delimiter='\t')
    cw.writerow([question + ' ?', answers])
    for i in range(50):
        expansion = t0_file[idx*50+i]
        query = question.strip() + ' ? ' + expansion.strip()
        cw.writerow([query, answers])
    fw.close()
    idx += 1
