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
    backend/linux-x86_64/gpu28CudaBackend \
    backend/linux-x86_64/gpu28CudaBackend-twophaseflux-validation \
    backend/SHA256SUMS \
    backend/BINARY-LICENSE.txt \
    backend/manifest.txt \
    CHANGELOG.md
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
grep -q '0x47553238U' gpu/GpuBackendProtocol.H \
    || fail 'GU28 compatibility magic is missing'
grep -q 'protocolMajor = 3' gpu/GpuBackendProtocol.H \
    || fail 'GU28 major revision mismatch'
grep -q 'protocolMinor = 2' gpu/GpuBackendProtocol.H \
    || fail 'GU28 minor revision mismatch'
grep -q '^version: 2[.]8[.]0$' CITATION.cff \
    || fail 'CITATION.cff version is not 2.8.0'

if ldd backend/linux-x86_64/gpu28CudaBackend \
    | grep -Eq 'libOpenFOAM|libfiniteVolume|libmeshTools|libcudart'
then
    fail 'production backend violates the binary dependency boundary'
fi
if ldd backend/linux-x86_64/gpu28CudaBackend-twophaseflux-validation \
    | grep -Eq 'libOpenFOAM|libfiniteVolume|libmeshTools|libcudart'
then
    fail 'validation backend violates the binary dependency boundary'
fi

find . -type f \
    \( -name 'Allrun' -o -name 'Allclean' -o -name 'Allwmake' \
       -o -name 'Allrestore' -o -name '*.sh' \) \
    -print0 | xargs -0 -n1 bash -n

fence_count="$(awk '/^```/{n++} END{print n+0}' README.md)"
(( fence_count % 2 == 0 )) || fail 'README.md has an unclosed code fence'

printf 'GPU-Riemann-GKP 2.8.0 release audit: PASS\n'
