#!/usr/bin/env python3
"""Read-only fluid/thermal production parity with exact, mutation-tested exceptions.

Usage: python3 tools/check_gas_parity.py --reference /path/to/ugkp-thermal
Inventories are read from BOTH roots. Joint local-only production inputs are
also compared; local_only is never a whole-header exemption. Thermal-only
applications and files absent from fluid are outside the gas-only scope.
"""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess

ROOT=Path(__file__).resolve().parents[1]
SUFFIXES={'.H','.h','.cuh','.cu','.C','.inl','.inc','.sh'}


def production(path):
    p=Path(path)
    return (p.suffix in SUFFIXES and (path.startswith('common/') or path.startswith('applications/gasUGKP/'))
        and not any(part in {'tests','devtools','build','build_logs','lnInclude','Make','__pycache__'} for part in p.parts))


def inventory(root):
    manifest=root/'tools/managed_mirrors.json'
    if not manifest.exists():return set()
    return {e['path'] for e in json.loads(manifest.read_text())['mirrors'] if production(e['path'])}


def observed(root):
    return {p.relative_to(root).as_posix() for scope in ('common','applications/gasUGKP')
        for p in (root/scope).rglob('*') if (p.is_file() or p.is_symlink()) and production(p.relative_to(root).as_posix())}


def normalized(path,text,rules):
    for rule in rules:
        if rule['path']!=path:continue
        old=rule['candidate']
        if text.count(old)>1:raise ValueError('ambiguous exception: '+path+' '+rule['reason'])
        if old in text:text=text.replace(old,rule['reference'],1)
    return text


def gas_transport(path):
    q=subprocess.run(['cpp','-P','-DGPU_OPERATOR_THERMAL=0',str(path)],capture_output=True,text=True)
    if q.returncode:raise ValueError('gas preprocessor failed: '+q.stderr)
    return q.stdout


def compare(candidate,reference):
    rules=json.loads((ROOT/'tools/gas_parity_allowances.json').read_text())['rules']
    shared=inventory(candidate)|inventory(reference)
    left,right=observed(candidate),observed(reference)
    paths=shared|(left&right)
    errors=[];checked=[]
    for relative in sorted(paths):
        a,b=candidate/relative,reference/relative
        if not a.is_file() or not b.is_file():
            errors.append('missing shared input: '+relative);continue
        try:
            if relative=='common/GpuParticleTransport.cuh':
                av,bv=gas_transport(a),gas_transport(b)
            else:
                av=normalized(relative,a.read_text(),rules);bv=b.read_text()
            if av!=bv:errors.append('production drift: '+relative)
            else:checked.append(relative)
        except (OSError,ValueError) as e:errors.append(str(e))
    # A new fluid-only production input cannot silently become an exemption.
    # The sole existing fluid-only module contains the research entry decoder.
    unexpected=(left-right)-shared-{'common/GpuResearchHierarchy.H'}
    errors += ['unclassified fluid-only production input: '+p for p in sorted(unexpected)]
    return dict(checked=len(checked),errors=errors,
        thermal_only_not_imported=sorted((right-left)-shared),
        exact_exception_paths=sorted({r['path'] for r in rules}),
        gas_transport_policy='GPU_OPERATOR_THERMAL=0 preprocessed equality')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--candidate',type=Path,default=ROOT)
    ap.add_argument('--reference',type=Path,required=True)
    ap.add_argument('--json',type=Path,help='Optional evidence output outside both checkouts')
    args=ap.parse_args();candidate=args.candidate.resolve();reference=args.reference.resolve()
    if args.json:
        destination=args.json.resolve()
        if any(destination==root or root in destination.parents for root in (candidate,reference)):
            ap.error('--json must be outside both source checkouts')
    result=compare(candidate,reference)
    if args.json:args.json.write_text(json.dumps(result,indent=2)+'\n')
    print('\n'.join(result['errors']) if result['errors'] else f"PASS {result['checked']} production inputs; exact research/compiler exceptions; gas-only transport equality")
    return int(bool(result['errors']))

if __name__=='__main__':raise SystemExit(main())
