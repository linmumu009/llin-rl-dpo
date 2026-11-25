"""Read-only SSH backup of this project's experiment evidence, without weights.

Run with the user's existing SSH credentials. The downloaded archive and extracted
files stay local until reviewed for publication. No remote files are modified.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile

REMOTE = r'''
import os,sys,json,hashlib,tarfile,io,subprocess,collections
roots={'project':'/data/liulin/llin-rl-dpo','lrsearch':'/data/liulin/llin-rl-dpo-lrsearch'}
allowed={'.json','.jsonl','.yaml','.yml','.toml','.ini','.cfg','.csv','.tsv','.txt','.md','.sh','.py','.patch','.diff','.log','.out','.err','.exit','.png','.svg','.pdf'}
binary={'.pt','.pth','.bin','.distcp','.safetensors','.ckpt','.onnx','.so','.whl','.arrow','.npy','.npz'}
prune={'.git','.venvs','.venv','venv','__pycache__','cache','.cache','node_modules','checkpoints'}
selected=[]; skipped=[]; frameworks=[]; containers=[]
def walk(root,label,framework=False):
 for base,ds,fs in os.walk(root,followlinks=False):
  ds[:]=sorted(d for d in ds if d not in prune and not os.path.islink(os.path.join(base,d)))
  rel=os.path.relpath(base,root)
  if rel=='.' and label=='project': ds[:]=[d for d in ds if not d.startswith('llin-paper')]
  if rel=='reference':
   for d in list(ds):
    if d!='rjx_repro':
     p=os.path.join(base,d)
     def git(*args):
      r=subprocess.run(['git','-C',p,*args],stdout=subprocess.PIPE,stderr=subprocess.DEVNULL)
      return r.stdout.decode(errors='replace').strip() if r.returncode==0 else None
     frameworks.append({'path':p,'commit':git('rev-parse','HEAD'),'remote':git('config','--get','remote.origin.url')})
     # Keep our custom configs, logs, data and launchers, not framework source.
     for b,dd,ff in os.walk(p):
      dd[:]=[x for x in dd if not x.startswith('.') and x not in prune]
      rr=os.path.relpath(b,p)
      if rr.count(os.sep)>3: dd[:]=[];continue
      custom=any(x.startswith('llin') for x in rr.split(os.sep))
      for f in ff:
       if custom or f.startswith(('run_llin','llin_')): add(os.path.join(b,f),label+'/'+os.path.relpath(os.path.join(b,f),root))
     ds.remove(d)
  for f in sorted(fs): add(os.path.join(base,f),label+'/'+os.path.relpath(os.path.join(base,f),root))
def add(p,name):
 if os.path.islink(p) or not os.path.isfile(p): return
 size=os.stat(p).st_size; ext=os.path.splitext(p)[1].lower(); f=os.path.basename(p)
 reason=None
 if ext in binary or f in {'.metadata','training_args.bin'}: reason='weights/optimizer/RNG/binary/regenerable cache'
 elif ext not in allowed and not f.startswith('events.out.tfevents'): reason='not experiment evidence allowlist'
 elif f.startswith(('.env','id_rsa','id_ed25519')) or f.endswith(('.pem','.key')) or 'SSH_PRIVATE' in f: reason='credentials'
 elif size>90*1024*1024: reason='over 90 MiB; review separately'
 if reason: skipped.append({'path':name,'bytes':size,'reason':reason}); return
 h=hashlib.sha256()
 with open(p,'rb') as stream:
  for block in iter(lambda:stream.read(1024*1024),b''): h.update(block)
 selected.append({'source':p,'path':name.replace(os.sep,'/'),'bytes':size,'sha256':h.hexdigest()})
for label,root in roots.items(): walk(root,label)
names=subprocess.check_output(['docker','ps','-a','--format','{{.Names}}']).decode().splitlines()
for name in names:
 if name not in {'llin-rl-dpo','llin-rl-dpo-lrsearch'} and not name.startswith('llin-msmm-'): continue
 c=json.loads(subprocess.check_output(['docker','inspect',name]))[0]
 if not any(m['Source'] in roots.values() for m in c.get('Mounts',[])): continue
 containers.append({'name':name,'image':c['Config']['Image'],'image_id':c['Image'],'state':c['State']['Status'],'mounts':c.get('Mounts',[])})
 upper=c['GraphDriver']['Data'].get('UpperDir','')
 w=upper+'/workspace'
 if not os.path.isdir(w): continue
 for b,ds,fs in os.walk(w):
  ds[:]=[d for d in ds if not d.startswith('.') and d not in prune]
  rel=os.path.relpath(b,w)
  if rel.count(os.sep)>5: ds[:]=[];continue
  custom=any(p.startswith('llin_') for p in rel.split(os.sep))
  for f in fs:
   if custom or f.startswith(('llin_','run_llin')): add(os.path.join(b,f),'containers/'+name+'/workspace/'+os.path.relpath(os.path.join(b,f),w))
manifest={'roots':roots,'files':selected,'excluded_files':skipped,'excluded_directories':sorted(prune)+['third-party framework source (custom llin files retained)','project/llin-paper* (separate project)'],'frameworks':frameworks,'containers':containers}
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as tar:
 for item in selected: tar.add(item['source'],arcname=item['path'],recursive=False)
 data=json.dumps(manifest,ensure_ascii=False,indent=2).encode()
 info=tarfile.TarInfo('MANIFEST.json');info.size=len(data);tar.addfile(info,io.BytesIO(data))
'''

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--destination', required=True)
    parser.add_argument('--host', required=True)
    parser.add_argument('--port', default='22205')
    parser.add_argument('--identity', required=True)
    args = parser.parse_args()
    dest = Path(args.destination).resolve()
    dest.mkdir(parents=True, exist_ok=True)
    archive = dest / 'download.tar.gz'
    command = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=20', '-p', args.port,
               '-i', args.identity, args.host, 'python3 -']
    with archive.open('wb') as stream:
        result = subprocess.run(command, input=REMOTE.encode(), stdout=stream)
    if result.returncode:
        raise SystemExit('SSH backup failed; partial archive retained for diagnosis')
    with tarfile.open(archive) as tar:
        tar.extractall(dest / 'raw', filter='data')
    manifest = json.loads((dest / 'raw/MANIFEST.json').read_text(encoding='utf-8'))
    for item in manifest['files']:
        path = dest / 'raw' / item['path']
        if hashlib.sha256(path.read_bytes()).hexdigest() != item['sha256']:
            raise SystemExit(f'Checksum mismatch: {item["path"]}')
    print(json.dumps({'verified_files': len(manifest['files']),
                      'original_bytes': sum(f['bytes'] for f in manifest['files']),
                      'archive_bytes': archive.stat().st_size,
                      'destination': str(dest)}, indent=2))

if __name__ == '__main__':
    main()
