from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import csv
import tqdm
import sys

tokenizer = AutoTokenizer.from_pretrained("bigscience/T0_3B")
model = AutoModelForSeq2SeqLM.from_pretrained("bigscience/T0_3B")
model.cuda()
num_per_q = 50

input_file = sys.argv[1]
output_file = sys.argv[2]

print("Start generating", input_file, flush=True)
fw = open(output_file, 'w')
cw = csv.writer(fw, delimiter='\t')
data = open(input_file, 'r').readlines()
for i in tqdm.trange(len(data)):
    q = data[i]
    inputs = tokenizer.encode(q.strip()+" ? To answer this question, we need to know", return_tensors="pt")
    outputs = model.generate(inputs.cuda(), max_new_tokens=100, do_sample=False, top_k=50)
    result = [tokenizer.decode(outputs[0], skip_special_tokens=True)]
    outputs = model.generate(inputs.cuda(), max_new_tokens=100, do_sample=True, top_k=50, num_return_sequences=num_per_q-1)
    result += [tokenizer.decode(outputs[j], skip_special_tokens=True) for j in range(num_per_q-1)]
    result = [str(i), q.strip()] + result
    cw.writerow(result)
    fw.flush()
fw.close()
