#!/usr/bin/env python3
"""Generate a new current-schema Taylor-Green input without touching existing cases."""
from pathlib import Path
import argparse,shutil,subprocess,sys
from migrate_case import migrate
ROOT=Path(__file__).resolve().parents[1]
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    parser.add_argument('--mesh',type=int,default=8)
    parser.add_argument('--parcels-per-cell',type=int,default=4)
    parser.add_argument('--steps',type=int,default=20)
    parser.add_argument('--delta-t',type=float,default=1e-5)
    args=parser.parse_args()
    if min(args.mesh,args.parcels_per_cell,args.steps)<1 or args.delta_t<=0:parser.error('positive mesh, parcels, steps and delta-t required')
    target=args.output.resolve()
    if target.exists():raise FileExistsError(target)
    template=ROOT/'legacy/study-cases/taylorGreen'
    for name in ('constant','system'):shutil.copytree(template/name,target/name)
    subprocess.run([sys.executable,str(template/'tools/prepare_case.py'),'--case',str(target),'--mesh',str(args.mesh),'--parcels-per-cell',str(args.parcels_per_cell),'--steps',str(args.steps),'--delta-t',str(args.delta_t),'--state','S1','--force-crossing-sentinels','--restart-format','binary'],check=True)
    migrate(target,'S1')
if __name__=='__main__':main()
