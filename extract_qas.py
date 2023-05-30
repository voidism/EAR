import json
import csv
import os
import sys
import tqdm

def extract_qas(json_file, csv_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
        with open(csv_file, 'w') as f:
            writer = csv.writer(f, delimiter='\t')
            for item in tqdm.tqdm(data):
                question = item['question']
                answers = item['answers']
                writer.writerow([question, answers])

if __name__ == "__main__":
    json_file = sys.argv[1]
    csv_file = sys.argv[2]
    extract_qas(json_file, csv_file)
