from pathlib import Path
from datasets import Dataset, concatenate_datasets
base = Path('/workspace/llin-rl-dpo/cache/rjx_cutoff_test_native/json/default-e7d1a27e35d3497b/0.0.0/95cd12cb79bdd40ece813ca76cc5f709067bfb719d56048dc7651d7403a4c3d2')
files = sorted(base.glob('cache-aa996564c506cfe1_*_of_00016.arrow'))
ds = concatenate_datasets([Dataset.from_file(str(p)) for p in files])
print('rank iteration mapping for first 3 iters: rank -> original indices, max_len')
for it in range(3):
    print('ITER', it + 1)
    for rank in range(8):
        idxs = [it * 16 + rank, it * 16 + rank + 8]
        lens = [len(ds[i]['input_ids']) for i in idxs]
        lab_counts = []
        for i in idxs:
            lab = ds[i]['labels']
            lab_counts.append(sum(1 for v in lab if v != -100))
        print(rank, idxs, lens, lab_counts, 'max_len', max(lens))