#!/usr/bin/env bash
set -euo pipefail

repo_root="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
cd "${repo_root}"

fail()
{
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

git diff --check

for required in \
    backend/linux-x86_64/gpu30CudaBackend \
    backend/linux-x86_64/gpu30CudaBackend-twophaseflux-validation \
    backend/SHA256SUMS \
    backend/BINARY-LICENSE.txt \
    backend/manifest.txt \
    CHANGELOG.md \
    docs/GPU30_UPGRADE_GUIDE.md \
    docs/PARAMETER_REFERENCE.md
do
    [[ -f "${required}" ]] || fail "missing release file: ${required}"
done

(
    cd backend
    sha256sum --check SHA256SUMS
)

mapfile -t publish_paths < <(git ls-files --cached --others --exclude-standard)
if printf '%s\n' "${publish_paths[@]}" \
    | grep -Eiq '(^|/)(private_backend|build_logs)(/|$)|GpuBackendServer[.]C$|[.](cu|cuh|o|a)$'
then
    printf '%s\n' "${publish_paths[@]}" \
        | grep -Ei '(^|/)(private_backend|build_logs)(/|$)|GpuBackendServer[.]C$|[.](cu|cuh|o|a)$' >&2
    fail 'private CUDA implementation or build products are present in the publish set'
fi

grep -qx 'EXE = $(FOAM_USER_APPBIN)/GpuGkp' Make/files \
    || fail 'public frontend executable name is not GpuGkp'
grep -q '0x47333030U' gpu/GpuBackendProtocol.H \
    || fail 'G300 compatibility magic is missing'
grep -q 'protocolMajor = 5' gpu/GpuBackendProtocol.H \
    || fail 'G300 major revision mismatch'
grep -q 'protocolMinor = 0' gpu/GpuBackendProtocol.H \
    || fail 'G300 minor revision mismatch'
grep -q 'fvSolutionDict.subDict("GpuGkp")' readGpuGasConfiguration.H \
    || fail 'canonical fvSolution/GpuGkp controls are missing'
if find examples -type f -name fvSolution -print0 \
    | xargs -0 -r grep -nE '^[[:space:]]*GPU(2_6|2_8|3_0)[[:space:]]*$'
then
    fail 'an example contains a release-numbered solver-control dictionary'
fi
grep -q '^version: 3[.]0[.]0$' CITATION.cff \
    || fail 'CITATION.cff version is not 3.0.0'
grep -q 'backend/linux-x86_64/gpu30CudaBackend binary' .gitattributes \
    || fail '.gitattributes does not identify the production GPU 3.0 backend as binary'
grep -q 'backend/linux-x86_64/gpu30CudaBackend-twophaseflux-validation binary' .gitattributes \
    || fail '.gitattributes does not identify the validation GPU 3.0 backend as binary'
if grep -Eq 'gpu28CudaBackend|GPU28_CUDA_BACKEND' .gitignore .gitattributes; then
    fail 'repository metadata still contains the retired GPU 2.8 backend identity'
fi

if ldd backend/linux-x86_64/gpu30CudaBackend \
    | grep -Eq 'libOpenFOAM|libfiniteVolume|libmeshTools|libcudart'
then
    fail 'production backend violates the binary dependency boundary'
fi
if ldd backend/linux-x86_64/gpu30CudaBackend-twophaseflux-validation \
    | grep -Eq 'libOpenFOAM|libfiniteVolume|libmeshTools|libcudart'
then
    fail 'validation backend violates the binary dependency boundary'
fi

for backend in \
    backend/linux-x86_64/gpu30CudaBackend \
    backend/linux-x86_64/gpu30CudaBackend-twophaseflux-validation
do
    file "${backend}" | grep -q 'stripped' \
        || fail "backend is not stripped: ${backend}"
    elf_listing="$(/usr/local/cuda/bin/cuobjdump -lelf "${backend}")"
    for target in sm_75 sm_80 sm_86 sm_89 sm_90; do
        grep -q "${target}" <<< "${elf_listing}" \
            || fail "backend is missing ${target}: ${backend}"
    done
    ptx_listing="$(/usr/local/cuda/bin/cuobjdump -lptx "${backend}")"
    grep -q 'ptx' <<< "${ptx_listing}" \
        || fail "backend is missing forward-compatible PTX: ${backend}"
done

if grep -RInE \
    'gpuCsrHeavyCellThreshold|gpuCsrHeavyTileParticles|gpuCsrHeavyWorkerBlocksPerSM' \
    examples --include=ugkwpProperties
then
    fail 'an example still contains a retired fixed heavy-cell control'
fi

if grep -RInE 'GPU28_CUDA_BACKEND|gpu28CudaBackend|GPU2[.]8' \
    examples install.sh README.md backend/manifest.txt
then
    fail 'a release-facing file still selects the 2.8 backend identity'
fi

if grep -RInE 'log[.]gpu3[.]0|runtime[.]gpu3[.]0|GPU 3[.]0 numerical' examples; then
    fail 'an example still exposes an internal release number in a runtime artifact or plot label'
fi

find . -type f \
    \( -name 'Allrun' -o -name 'Allclean' -o -name 'Allwmake' \
       -o -name 'Allrestore' -o -name '*.sh' \) \
    -print0 | xargs -0 -n1 bash -n

fence_count="$(awk '/^```/{n++} END{print n+0}' README.md)"
(( fence_count % 2 == 0 )) || fail 'README.md has an unclosed code fence'

printf 'GPU-Riemann-GKP 3.0.0 release audit: PASS\n'
