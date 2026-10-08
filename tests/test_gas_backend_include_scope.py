"""Compile the backend's real include boundary without launching GPU work."""
from pathlib import Path
import shutil
import subprocess
import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_gas_backend_namespace_dependencies_compile_with_nvcc(tmp_path):
    nvcc = shutil.which('nvcc')
    if not nvcc:
        pytest.skip('nvcc unavailable; CUDA compilation not verified')
    backend = ROOT/'applications/gasUGKP/private_backend'
    source = (backend/'GpuResidentStrict.cu').read_text()
    # Retain every production global include and then the operator's namespace
    # dependency. A namespace-owning header first loaded inside this anonymous
    # namespace shadows ::ugkwp and reproduces the real integration failure.
    prefix = source.split('\nnamespace\n{', 1)[0]
    probe = tmp_path/'include_scope.cu'
    probe.write_text(prefix+'\nnamespace {\n#include "gasTransport/GasGeometryValidation.H"\n#include "gasTransport/SpeciesDiffusion.H"\n}\n')
    q = subprocess.run([nvcc, '-std=c++17', '-arch=sm_89', '-I'+str(backend),
        '-I'+str(backend.parent/'gpu'), '-I'+str(ROOT/'common'),
        '-c',str(probe),'-o',str(tmp_path/'include_scope.o')],capture_output=True,text=True)
    assert q.returncode == 0, q.stdout+q.stderr


def test_gas_occupancy_kernel_addresses_are_fully_specialized(tmp_path):
    import re
    nvcc = shutil.which('nvcc')
    if not nvcc:
        pytest.skip('nvcc unavailable; CUDA compilation not verified')
    backend = ROOT/'applications/gasUGKP/private_backend/GpuResidentStrict.cu'
    sources = [(ROOT/'common/GpuToolB1.cuh').read_text(), backend.read_text()]
    references = []
    for source in sources:
        references += re.findall(r'^\s*(recoverGasPrimitivesKernel(?:<[^>]+>)?|computeGasInternalFaceFluxKernel<[^>]+>),', source, re.M)
    assert len(references) == 4, references
    declarations = []
    for path,name in [('computeSstFaceFluxKernel.cuh','recoverGasPrimitivesKernel'),
                      ('computeGasInternalFaceFluxKernel.cuh','computeGasInternalFaceFluxKernel')]:
        text = (ROOT/'common/operators'/path).read_text()
        pattern = r'template<[^>]+>\s*__global__ void '+name+r'\([^)]*\)'
        declaration = re.search(pattern,text)
        assert declaration, name
        declarations.append(declaration[0]+' {}')
    probe = tmp_path/'occupancy.cu'
    probe.write_text('#include <cuda_runtime.h>\n#define GPU_OPERATOR_TIME double\nstruct DeviceState {};\n'
        +'\n'.join(declarations)+'\nvoid probe() { int count;\n'
        +'\n'.join('cudaOccupancyMaxActiveBlocksPerMultiprocessor(&count,'+ref+',128,0);' for ref in references)+'\n}\n')
    q = subprocess.run([nvcc,'-std=c++17','-arch=sm_89','-c',str(probe),'-o',str(tmp_path/'occupancy.o')],capture_output=True,text=True)
    assert q.returncode == 0,q.stdout+q.stderr
