#!/usr/bin/env python3
"""Explicit, in-place conversion of a COPIED historical fluid case to schema 1."""
from pathlib import Path
import argparse
import re
import shutil

# OpenFOAM Foundation 10 default RR = 1000 * NA * k (etc/controlDict).
OPENFOAM10_RR = 8314.47006650545

STATES = ('L0', 'L1', 'T1', 'E1', 'S1', 'T2', 'E2', 'S2')
SCHEDULE = {
    'gpuResidentPureGasOnly': 'false', 'gpuResidentDynamicInlet': 'false',
    'gpuResidentParticleCapacity': '1000000', 'gpuResidentMaxFaceWalkHops': '32',
    'gpuResidentCourantUpdateInterval': '1000', 'gpuResidentMaxDeltaTGrowth': '1.05',
}

def clean(text):
    return re.sub(r'//[^\n]*|/\*.*?\*/', '', text, flags=re.S)

def entry(text, name, default=None):
    match = re.search(r'(?<!\S)' + re.escape(name) + r'\s+([^;{}]+);', clean(text))
    if match:
        return match.group(1).strip()
    if default is None:
        raise ValueError(f'missing entry {name}')
    return default

def header(name):
    return f'FoamFile {{ version 2.0; format ascii; class dictionary; object {name}; }}\n'

def body(text):
    text = clean(text)
    return re.sub(r'FoamFile\s*\{[^}]*\}', '', text, count=1).strip()

def convert_restart(path):
    translations = {
        b'GPU2_PARTICLES_V4': b'UGKP_PARTICLES_SCHEMA4',
        b'GPU2_PARTICLES_V5_BIN': b'UGKP_PARTICLES_SCHEMA5_BIN',
        b'GPU3_PARTICLES_V1_BIN': b'UGKP_PARTICLES_SCHEMA1_BIN',
        b'GPU2_SOURCE_RESIDUAL_V1': b'UGKP_SOURCE_RESIDUAL_SCHEMA1',
    }
    with path.open('rb') as source:
        line = source.readline()
        kind = line.split()[0]
        if kind.startswith(b'UGKP_'):
            return
        if kind not in translations:
            raise ValueError(f'unsupported restart header {kind!r}: {path}')
        temporary = path.with_name(path.name + '.converting')
        with temporary.open('wb') as target:
            target.write(line.replace(kind, translations[kind], 1))
            shutil.copyfileobj(source, target, 8 << 20)
    temporary.replace(path)

def migrate(case, state):
    if state not in STATES:
        raise ValueError(f'unsupported research state {state}')
    case = Path(case)
    constant = case / 'constant'
    if (constant / 'fluidProperties').exists():
        raise FileExistsError('case already uses fluidProperties; refusing to overwrite')
    particle = (constant / 'ugkwpProperties').read_text()
    control = case / 'system/controlDict'
    control_text = control.read_text()
    if not re.search(r'\bapplication\s+[^;]+;', clean(control_text)):
        raise ValueError('controlDict is missing application')
    drag = entry(particle, 'dragModel', entry(particle, 'gpuResidentDragModel', 'SchillerNaumann'))
    heat = entry(particle, 'particleGasHeatTransferModel', entry(particle, 'gpuResidentHeatTransferModel', 'RanzMarshall'))
    if drag not in ('none','SchillerNaumann','GidaspowErgunWenYu') or heat not in ('none','RanzMarshall'):
        raise ValueError('adapter-only or unsupported particle model; use the dedicated validation solver')
    restarts = [path for name in ('gpuResidentStrictParticles.dat','gpuResidentStrictSourceResidual.dat') for path in case.glob('*/'+name)]
    supported_headers = {b'GPU2_PARTICLES_V4', b'GPU2_PARTICLES_V5_BIN', b'GPU3_PARTICLES_V1_BIN',
        b'GPU2_SOURCE_RESIDUAL_V1', b'UGKP_PARTICLES_SCHEMA4', b'UGKP_PARTICLES_SCHEMA5_BIN',
        b'UGKP_PARTICLES_SCHEMA1_BIN', b'UGKP_SOURCE_RESIDUAL_SCHEMA1'}
    for path in restarts:
        with path.open('rb') as stream:
            fields = stream.readline().split()
        if not fields or fields[0] not in supported_headers:
            raise ValueError(f'unsupported restart header: {path}')

    gks_path = constant / 'gksProperties'
    gks = gks_path.read_text() if gks_path.exists() else ''
    physical = constant / 'physicalProperties'
    if physical.exists():
        thermo = body(physical.read_text())
        momentum = constant / 'momentumTransport'
        turbulence = body(momentum.read_text()) if momentum.exists() else 'simulationType laminar;'
    else:
        gamma, gas_r = float(entry(gks, 'gamma')), float(entry(gks, 'R'))
        if entry(gks, 'model', 'laminar') != 'laminar':
            raise ValueError('legacy gks turbulence requires an explicit model conversion')
        thermo = ('thermoType { type hePsiThermo; mixture pureMixture; transport const; '
                  'thermo hConst; equationOfState perfectGas; specie specie; energy sensibleInternalEnergy; }\n'
                  f'mixture {{ specie {{ molWeight {OPENFOAM10_RR/gas_r:.17g}; }} '
                  f'thermodynamics {{ Cp {gamma*gas_r/(gamma-1):.12g}; Hf 0; }} '
                  f'transport {{ mu {entry(gks,"mu")}; Pr {entry(gks,"Pr")}; }} }}\n')
        turbulence = 'simulationType laminar;'
    fluid = header('fluidProperties') + 'schemaVersion 1;\n' + thermo + '\nturbulence\n{\n' + turbulence + '\n}\n'
    schedule = header('schedulingProperties') + 'schemaVersion 1;\ngpuPrecision 64;\n'
    for key, default in SCHEDULE.items():
        schedule += f'{key} {entry(particle,key,default)};\n'
    threads = str(1 << int(entry(particle, 'bn', '7')))
    level = 'L0' if state == 'L0' else ('L2' if state in ('T2','E2','S2') else 'L1')
    schedule += f'gpuCsrLevel {level};\ngpuResearchVariant {state};\ngpuParticleBlockThreads {threads};\ngpuReductionBlockThreads {threads};\n'
    particle_body = body(particle)
    removed = list(SCHEDULE) + ['gpuResidentStrict','bn','gpuCsrCellLocalPath','gpuCsrWarpAggregatedBinning',
        'gpuCsrHeavyReduction','gpuCsrSplitPreDirectory','gpuCsrHeavyCellThreshold','gpuCsrHeavyTileParticles',
        'gpuCsrHeavyWorkerBlocksPerSM','gpuCsrLevel','gpuResidentParticleResponseTime','gpuResidentParticleThermalResponseTime']
    for key in removed:
        particle_body = re.sub(r'(?m)^\s*'+re.escape(key)+r'\s+[^;]+;\s*', '', particle_body)
    particle_body = particle_body.replace('gpuResidentDragModel ', 'dragModel ').replace('gpuResidentHeatTransferModel ', 'particleGasHeatTransferModel ')
    for key in ('particleTemperatureTransport','particleRho','particleCp','particleTMin','particleTMax'):
        if re.search(r'\b'+key+r'\s', particle_body) is None:
            value = entry(gks, key, '')
            if value:
                particle_body += f'\n{key} {value};'
    schemes_path = case / 'system/fvSchemes'
    schemes = schemes_path.read_text()
    solution_path = case / 'system/fvSolution'
    solution = solution_path.read_text()
    if not re.search(r'\bfluxScheme\s+', clean(schemes)):
        fluxes={'hllKurganov':'Kurganov','hllc':'HLLC','hlle':'HLLE','roe':'Roe','hllem':'HLLEM','rusanov':'Tadmor','slau2':'SLAU2','slau2_2':'SLAU2.2'}
        flux=entry(gks,'flux'); flux=fluxes.get(flux,flux)
        reconstruction=entry(gks,'reconstruction','firstOrder')
        if reconstruction not in ('firstOrder','MUSCL','energyLimitedLinear'):
            raise ValueError(f'unsupported reconstruction {reconstruction}')
        u='MUSCL' if reconstruction=='MUSCL' else 'upwind'
        e='limitedLinear 1' if reconstruction=='energyLimitedLinear' else u
        schemes=header('fvSchemes')+f'fluxScheme {flux};\ngasLimiter {entry(gks,"limiter","none")};\n'
        schemes+=f'ddtSchemes {{ default {entry(gks,"timeIntegrator","Euler")}; }}\ngradSchemes {{ default Gauss linear; }}\ndivSchemes {{ default none;\n'
        schemes+=f'div(phi,U) Gauss {u};\n'
        for key in ('div(phi,e)','div(phi,K)','div(phi,(p|rho))'):
            schemes+=f'{key} Gauss {e};\n'
        schemes+='div(((rho*nuEff)*dev2(T(grad(U))))) Gauss linear;\n}\nlaplacianSchemes { default Gauss linear corrected; }\ninterpolationSchemes { default linear; }\nsnGradSchemes { default corrected; }\n'
        solution += '\nUGKP\n{\n'+f'rhoMin {entry(gks,"rhoMin","1e-12")};\nrobustFallback {entry(gks,"robustFallback","true")};\nmaxDiffusionNumber {entry(gks,"maxDiffusionNumber","0.25")};\n'+'}\n'
    else:
        solution = re.sub(r'\b(?:GpuGkp|GPU2_8|GPU2_6)\b', 'UGKP', solution)
    # Only mutate after all scalar/dictionary validation has completed.
    (constant/'fluidProperties').write_text(fluid)
    (constant/'particleProperties').write_text(header('particleProperties')+'schemaVersion 1;\n'+particle_body+'\n')
    (constant/'schedulingProperties').write_text(schedule)
    schemes_path.write_text(schemes)
    solution_path.write_text(solution)
    control.write_text(re.sub(r'\bapplication\s+[^;]+;', 'application gasUGKP;', control_text,count=1))
    for path in restarts:
        convert_restart(path)
    # Keep old dictionaries as migration evidence; the new reader uses schema-1 files only.

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case',type=Path)
    parser.add_argument('--state',choices=STATES,required=True)
    args=parser.parse_args()
    migrate(args.case,args.state)
