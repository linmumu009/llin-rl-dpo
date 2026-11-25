from pathlib import Path
from datasets import Dataset, concatenate_datasets
base = Path('/workspace/llin-rl-dpo/cache/rjx_cutoff_test_native/json/default-e7d1a27e35d3497b/0.0.0/95cd12cb79bdd40ece813ca76cc5f709067bfb719d56048dc7651d7403a4c3d2')
for prefix in ['cache-2cdf4eb21bc50869', 'cache-aa996564c506cfe1']:
    files = sorted(base.glob(f'{prefix}_*_of_00016.arrow'))
    print('\nPREFIX', prefix, 'files', len(files))
    dsets=[]
    for p in files:
        ds=Dataset.from_file(str(p))
        print(p.name, len(ds), ds.column_names)
        dsets.append(ds)
    if not dsets:
        continue
    ds=concatenate_datasets(dsets)
    print('TOTAL', len(ds), 'cols', ds.column_names)
    if 'input_ids' not in ds.column_names:
        continue
    rows=[]
    for i in range(min(40, len(ds))):
        inp=ds[i]['input_ids']
        lab=ds[i].get('labels')
        labels_non=-1
        first=last=None
        if lab is not None:
            idx=[j for j,v in enumerate(lab) if v!=-100]
            labels_non=len(idx)
            if idx:
                first, last = idx[0], idx[-1]
        rows.append((i, len(inp), labels_non, first, last))
    print('first40 idx/input_len/nonignore/first_label/last_label')
    for r in rows:
        print(r)
    lens=[len(ds[i]['input_ids']) for i in range(len(ds))]
    print('len min/max/count4096', min(lens), max(lens), sum(1 for x in lens if x==4096))
    if 'labels' in ds.column_names:
        counts=[]
        lasts=[]
        for i in range(len(ds)):
            lab=ds[i]['labels']
            idx=[j for j,v in enumerate(lab) if v!=-100]
            counts.append(len(idx))
            lasts.append(idx[-1] if idx else None)
        print('label nonignore min/max', min(counts), max(counts), 'last max', max(x for x in lasts if x is not None))