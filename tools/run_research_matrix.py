#!/usr/bin/env python3
"""Run paired research states from one immutable seed; reject invalid CUDA timings."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
FLAGS = {'L0': (0,0), 'L1': (1,0), 'T1': (1,0), 'S1': (1,0), 'T2': (1,1), 'S2': (1,1)}

def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(8<<20),b''):digest.update(chunk)
    return digest.hexdigest()

def validate_rows(rows, particle_path, heavy):
    if not rows:
        raise ValueError('no CUDA probe rows; rebuild with UGKP_DEVELOPMENT_PROBES=1')
    stages=[key for key in rows[0] if key.endswith('_ms')]
    for row in rows:
        if row['status']!='ok' or row['timing_valid']!='1':raise ValueError('invalid timing status')
        if int(row['particle_path'])!=particle_path or int(row['heavy_reduction_enabled'])!=heavy:
            raise ValueError('wrong research execution path')
        if int(row['base_particle_count'])+int(row['injected_particle_count'])!=int(row['pretransport_particle_count']):
            raise ValueError('injection ledger mismatch')
        if int(row['pretransport_particle_count'])-int(row['removed_particle_count'])!=int(row['particle_count']):
            raise ValueError('removal ledger mismatch')
        if row['occupancy_matches_count']!='1' or int(row['occupancy_sum'])!=int(row['particle_count']):
            raise ValueError('occupancy mismatch')
        if any(int(row[key],0)!=0 for key in ('bad_cells','bad_particles','bad_field_mask')):
            raise ValueError('invalid physical state')
        if any(not math.isfinite(float(row[key])) or float(row[key])<0 for key in stages):
            raise ValueError('invalid CUDA time')
    return {key:statistics.median(float(row[key]) for row in rows) for key in stages}

def replace(text,key,value):
    text,count=re.subn(r'(?m)^(\s*'+re.escape(key)+r'\s+)[^;]+;',lambda m:m[1]+str(value)+';',text,count=1)
    if count!=1:raise ValueError(f'missing control {key}')
    return text

def field(path):
    import numpy as np
    text=path.read_text()
    match=re.search(r'internalField\s+nonuniform\s+List<\w+>\s+(\d+)\s*\((.*?)\)\s*;',text,re.S)
    if match:
        data=np.fromstring(re.sub('[()]',' ',match[2]),sep=' ')
    else:
        match=re.search(r'internalField\s+uniform\s+([^;]+);',text)
        if not match:raise ValueError(f'missing field {path}')
        data=np.fromstring(re.sub('[()]',' ',match[1]),sep=' ')
    if not len(data) or not np.isfinite(data).all():raise ValueError(f'nonfinite or empty {path}')
    return data

def times(case):
    result=[]
    for p in case.iterdir():
        if p.is_dir():
            try:result.append((float(p.name),p))
            except ValueError:pass
    return sorted(result)

def run(args):
    import numpy as np
    from migrate_case import migrate
    seed=args.case.resolve();out=args.output.resolve()
    if out.exists():raise FileExistsError(out)
    if out.is_relative_to(seed):raise ValueError('output must be outside the seed case')
    if not args.states or len(set(args.states))!=len(args.states):raise ValueError('states must be unique and nonempty')
    if args.steps<1 or args.repeats<1:raise ValueError('steps and repeats must be positive')
    frontend=ROOT/'build/bin/gasUGKP';backend=ROOT/'build/bin/gasUGKPCudaBackend'
    initial=times(seed)[0][1]
    restart=initial/'gpuResidentStrictParticles.dat'
    identity={'seed':str(seed),'restart_sha256':sha(restart),'frontend_sha256':sha(frontend),'backend_sha256':sha(backend),
        'steps':args.steps,'repeats':args.repeats,'states':args.states,'warmup_processes_per_state':1,
        'purpose':'current-kernel validation; historical paper timings are not replaced'}
    out.mkdir(parents=True)
    (out/'manifest.json').write_text(json.dumps(identity,indent=2))
    reference=None;summary=[];comparisons=[]
    for repetition in range(args.repeats+1):
        # Alternating order reduces a fixed order/temperature bias.
        order=args.states if repetition%2==0 else list(reversed(args.states))
        for state in order:
            run_id=f'r{repetition}_{state}';folder=out/run_id;case=folder/'case'
            folder.mkdir()
            for name in ('constant','system',initial.name):shutil.copytree(seed/name,case/name)
            if (case/'constant/fluidProperties').exists():
                schedule=case/'constant/schedulingProperties';text=schedule.read_text()
                text=re.sub(r'(?m)^\s*gpuResearchVariant\s+[^;]+;\s*','',text)
                text=replace(text,'gpuCsrLevel','L0' if state=='L0' else ('L2' if FLAGS[state][1] else 'L1'))
                schedule.write_text(text+f'\ngpuResearchVariant {state};\n')
            else:migrate(case,state)
            control=case/'system/controlDict';text=control.read_text()
            dt=float(re.search(r'(?m)^\s*deltaT\s+([^;]+);',text)[1])
            for key,value in [('application','gasUGKP'),('startFrom','startTime'),('startTime',initial.name),
                ('endTime',format(float(initial.name)+args.steps*dt,'.15g')),('adjustTimeStep','false'),
                ('writeControl','timeStep'),('writeInterval',args.steps)]:
                text=replace(text,key,value)
            text=replace(text,'timePrecision',16)
            control.write_text(text)
            # Archived time indices need not be divisible by this short gate's write interval.
            time_metadata=case/initial.name/'uniform/time'
            if time_metadata.exists():
                metadata=time_metadata.read_text()
                for key,value in [('index',0),('value',initial.name),('deltaT',dt),('deltaT0',dt)]:
                    metadata=replace(metadata,key,value)
                time_metadata.write_text(metadata)
            if not (case/'constant/polyMesh/points').exists():
                with (folder/'mesh.log').open('w') as log:
                    subprocess.run(['blockMesh','-case',str(case)],stdout=log,stderr=subprocess.STDOUT,check=True)
            env=dict(os.environ,GAS_UGKP_CUDA_BACKEND=str(backend),UGKP_DEV_PROBE_MODE='full',
                UGKP_DEV_PROBE_LOG=str(folder/'stage_timing.csv'),UGKP_DEV_PROBE_INTERVAL='1',
                UGKP_DEV_PROBE_RUN_ID=run_id,UGKP_DEV_PROBE_VARIANT=state,UGKP_DEV_PROBE_FAIL_ON_NONFINITE='1')
            started=time.time()
            with (folder/'solver.log').open('w') as log:
                subprocess.run([str(frontend),'-case',str(case)],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            with (folder/'stage_timing.csv').open() as stream:rows=list(csv.DictReader(stream))
            if len(rows)!=args.steps or any(row.get('run_id')!=run_id for row in rows):raise ValueError('probe length/run identity mismatch')
            stats=validate_rows(rows,1,FLAGS[state][1])
            expected_flags={"L0":(0,0,0,0),"L1":(1,0,0,0),"T1":(1,1,0,0),"S1":(1,1,1,0),"T2":(1,1,0,1),"S2":(1,1,1,1)}[state]
            expected=f"Research hierarchy: {state} cell={expected_flags[0]} warp={expected_flags[1]} split={expected_flags[2]} heavy={expected_flags[3]}"
            if expected not in (folder/"solver.log").read_text():raise ValueError("wrong configured research flags")
            final_time,final=times(case)[-1]
            if not math.isclose(final_time,float(initial.name)+args.steps*dt,abs_tol=dt*1e-4):raise ValueError('wrong terminal time')
            values={name:field(final/name) for name in ('rho','p','T','U','epsilonS','Us','theta')}
            if any((values[key]<=0).any() for key in ('rho','p','T')):raise ValueError('nonpositive gas state')
            errors={}
            if reference is None:reference=values
            for name,value in values.items():
                errors[name]=float(np.max(np.abs(value-reference[name]))/max(1.0,float(np.max(np.abs(reference[name])))))
                if errors[name]>args.field_tolerance:raise ValueError(f'{run_id}: {name} differs from paired baseline: {errors[name]}')
            comparisons.append({'run_id':run_id,'relative_linf_errors':errors})
            stats.update(run_id=run_id,state=state,repetition=repetition,measured=int(repetition>0),particle_count=int(rows[-1]['particle_count']),
                heavy_task_count=int(rows[-1].get('heavy_task_count_estimate','0')),wall_seconds=time.time()-started)
            (folder/'validation.json').write_text(json.dumps(dict(passed=True,**stats,field_errors=errors),indent=2))
            summary.append(stats)
            with (out/'summary.csv').open('w') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
            print(f'{run_id}: PASS, particles={stats["particle_count"]}, CUDA total={stats["total_ms"]:.6g} ms',flush=True)
    (out/'field_comparison.json').write_text(json.dumps(comparisons,indent=2))
    (out/'PASS.json').write_text(json.dumps({'passed':True,'processes':len(summary),'max_relative_field_error':max(max(x['relative_linf_errors'].values()) for x in comparisons)},indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case',type=Path);parser.add_argument('output',type=Path)
    parser.add_argument('--states',nargs='+',choices=list(FLAGS),default=list(FLAGS))
    parser.add_argument('--steps',type=int,default=20);parser.add_argument('--repeats',type=int,default=1)
    parser.add_argument('--field-tolerance',type=float,default=1e-8)
    run(parser.parse_args())
