"""Instantiate actual common gas/SST CUDA kernels in FP32/FP64; never execute."""
from pathlib import Path
import re
import shutil
import subprocess
import pytest
ROOT=Path(__file__).resolve().parents[1]

@pytest.mark.parametrize('bits',[32,64])
def test_actual_shared_gas_kernels_compile_with_nvcc(tmp_path,bits):
    nvcc=shutil.which('nvcc')
    if not nvcc:pytest.skip('nvcc unavailable; native shared-operator compilation not verified')
    # Reuse only the production include list and type adapters from the host
    # probe. Remove every CUDA emulator macro/index and keep real CUDA syntax.
    text=(ROOT/'tests/gas_transport/gas_state_probe.cpp').read_text()
    prefix=text.split('#ifdef LEGACY_OPTIONAL_SPECIES',1)[0]
    prefix=re.sub(r'^#define (?:__host__|__global__|__device__|__forceinline__|asm)\b[^\n]*\n','',prefix,flags=re.M)
    prefix=re.sub(r'^struct ThreadIndex[^\n]*\n','',prefix,flags=re.M)
    kernels=[
      ('initialiseSstConservativeStateKernel',''),('saveGasConservativeStateKernel',''),
      ('recoverGasPrimitivesKernel',''),('recoverSstPrimitivesKernel',''),
      ('updateLegacyGasBoundaryMirrorKernel',',0.0'),('updateRiemannBoundaryMirrorKernel',''),
      ('updateWaveTransmissivePressureBoundaryKernel',',1e-6'),('applySstWallFunctionStateKernel',''),
      ('computeGasHllcAdcSensorKernel',''),('computeGasPrimitiveGradientsKernel',''),
      ('computeSstGradientsKernel',''),('computeGasGradientLimiterKernel',''),('computeGasEddyViscosityKernel',''),
      ('computeGasInternalFaceFluxKernel<true>',',1e-6'),('computeGasInternalFaceFluxKernel<false>',',1e-6'),
      ('enforcePeriodicGasFluxAntisymmetryKernel',''),('computeGasFluxPositivityScaleKernel',',1e-6'),
      ('applyGasFluxPositivityScaleKernel',''),('computeSstFaceFluxKernel',''),('enforcePeriodicSstFluxAntisymmetryKernel',''),
      ('applySstFluxAndSourceKernel',',1e-6'),('applyGasFluxDivergenceByCellKernel',',1e-6'),
      ('blendGasConservativeStateKernel',',Real(.75),Real(.25)'),
      ('computeGasDiffusionNumberKernel',',1e-6,.5'),('computeSstStabilityNumberKernel',',1e-6,.5'),
      ('computeGasCourantFieldKernel',',1e-6'),('computeGasConvectiveCourantByCellKernel',',1e-6')]
    body='\nusing GasView=ugkwp::GasStateView<Real,Time,ugkwp::SstCoefficients>;\n'
    body+='struct MixtureView:GasView {ugkwp::GasSpeciesState<Real,2> gasSpecies;};\n'
    body+='template<class State> void instantiate(State* s) {\n'
    body+='\n'.join(name+'<<<1,32>>>(s'+args+');' for name,args in kernels)+'\n}\n'
    body+='template void instantiate<GasView>(GasView*);\ntemplate void instantiate<MixtureView>(MixtureView*);\n'
    source=tmp_path/f'gas{bits}.cu';source.write_text('#include <cuda_runtime.h>\n'+prefix+body)
    q=subprocess.run([nvcc,'-std=c++17','-arch=sm_89','-O1',f'-DUGKWP_GPU_REAL_BITS={bits}',
        '-I'+str(ROOT/'common'),'-I'+str(ROOT/'common/gasNumerics'),'-I'+str(ROOT/'tests/gas_transport'),
        '-c',str(source),'-o',str(tmp_path/f'gas{bits}.o')],capture_output=True,text=True)
    assert q.returncode==0,q.stdout+q.stderr
    assert (tmp_path/f'gas{bits}.o').stat().st_size>0
