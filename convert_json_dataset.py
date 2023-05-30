
import json
import sys
import tqdm
import glob
'''Converting nq-test-xxx/results.json to csv file for training a RoBERTa model.'''

# [print(j[i]['question'], j[i]['hit_min_rank']) for i in range(51)]

def process_worker(results_dir):
    with open(results_dir, 'r') as f:
        data = json.load(f)
        key = results_dir.split('/')[-2]
        results = []
        for d in data[1:]:
            results.append([d['question'], d['hit_min_rank'], d["ctxs"][0]["title"], d["ctxs"][0]["text"]])
    return key, results


def main():
    if len(sys.argv) < 5:
        print('Usage: python convert_json_to_trainset.py <output_dir> <results_dir_glob pattern> <n_workers> <n_examples>')
        exit(0)

    output_dir = sys.argv[1]
    n_examples = int(sys.argv[4])
    results_dirs = [sys.argv[2]%(i) for i in range(n_examples)] # 79168 for train 3610 for test
    n_workers = int(sys.argv[3])
    all_data = {}
    if n_workers == 1:
        for results_dir in tqdm.tqdm(results_dirs):
            with open(results_dir, 'r') as f:
                data = json.load(f)
                key = results_dir.split('/')[-2]
                results = []
                for d in data[1:]:
                    results.append([d['question'], d['hit_min_rank'], d["ctxs"][0]["title"], d["ctxs"][0]["text"]]) # if d['hit_min_rank'] is not None else 101])
                all_data[key] = results
    else:
        import multiprocessing as mp
        pool = mp.Pool(n_workers)
        all_data_list = pool.map(process_worker, results_dirs)
        # for results_dir in tqdm.tqdm(results_dirs):
        #     pool.apply_async(process_worker, args=(results_dir, all_data))
        pool.close()
        pool.join()
        all_data = dict(all_data_list)
    
    # Save the results
    with open(output_dir, 'w') as f:
        json.dump(all_data, f, indent=4)

if __name__ == '__main__':
    main()


