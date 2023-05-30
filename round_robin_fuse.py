import json
import sys
import tqdm
'''Round robin fuse the results of the 1st and 2nd (or more) best models.'''

def main():
    if len(sys.argv) < 2:
        print('Usage: python round_robin_fuse.py <output_dir> <results_dir> ... <results_dir_n>')
        exit(0)

    output_dir = sys.argv[1]
    results_dirs = sys.argv[2:]
    results = []
    for results_dir in results_dirs:
        with open(results_dir, 'r') as f:
            results.append(json.load(f))
    assert len(results) > 0
    for result in results[1:]:
        assert len(result) == len(results[0])

    # Fuse the results
    fused_results = []
    for i in tqdm.trange(len(results[0])): # 3610 questions
        fused_result = {"question": results[0][i]["question"], "answers": results[0][i]["answers"], "ctxs": [], "hit_min_rank": None, "all_hits": []}
        rank = 1
        added = {}
        for j in range(len(results[0][i]["ctxs"])): # 100 contexts
            for result in results: # 2 results
                if result[i]["ctxs"][j]["id"] not in added:
                    fused_ctx = {"id": result[i]["ctxs"][j]["id"], "rank": rank, "score": result[i]["ctxs"][j]["score"], "text": result[i]["ctxs"][j]["text"], "title": result[i]["ctxs"][j]["title"], "has_answer": result[i]["ctxs"][j]["has_answer"]}
                    if fused_ctx['has_answer']:
                        fused_result["hit_min_rank"] = min(fused_result["hit_min_rank"], rank) if fused_result["hit_min_rank"] is not None else rank
                        fused_result["all_hits"].append(rank)
                    fused_result['ctxs'].append(fused_ctx)
                    added[fused_ctx["id"]] = True
                    rank += 1
            if rank > 100:
                break
        fused_results.append(fused_result)

    # Save the results
    with open(output_dir, 'w') as f:
        json.dump(fused_results, f, indent=4)

if __name__ == '__main__':
    main()
