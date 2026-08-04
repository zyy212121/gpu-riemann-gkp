#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${repo_root}/scripts/openfoam10-env.sh"

backend_source="${repo_root}/backend/linux-x86_64/gpu26CudaBackend"
backend_target="${FOAM_USER_APPBIN}/gpu26CudaBackend"
frontend_target="${FOAM_USER_APPBIN}/GpuGkp"

if [[ "$(uname -s)" != "Linux" || "$(uname -m)" != "x86_64" ]]; then
    printf 'ERROR: the supplied CUDA backend requires Linux x86_64.\n' >&2
    exit 2
fi

if [[ ! -f "${backend_source}" ]]; then
    printf 'ERROR: packaged backend is missing: %s\n' "${backend_source}" >&2
    exit 2
fi

(
    cd "${repo_root}/backend"
    sha256sum --check --status SHA256SUMS
) || {
    printf 'ERROR: packaged backend checksum verification failed.\n' >&2
    exit 2
}

mkdir -p "${FOAM_USER_APPBIN}"
install -m 0755 "${backend_source}" "${backend_target}"

rm -f \
    "${repo_root}/Make/${WM_OPTIONS}/diluteUgkwpFoam.o" \
    "${repo_root}/Make/${WM_OPTIONS}/gpu/GpuBackendClient.o" \
    "${repo_root}/Make/${WM_OPTIONS}/diluteUgkwpFoam.C.dep" \
    "${repo_root}/Make/${WM_OPTIONS}/gpu/GpuBackendClient.C.dep"

(
    cd "${repo_root}"
    UGKWP_CUDA_EXE_INC="-DUGKWP_USE_CUDA" wmake
)

test -x "${frontend_target}"
test -x "${backend_target}"

if ldd "${frontend_target}" | grep -Eq 'libcuda|libcudart'; then
    printf 'ERROR: the OpenFOAM frontend unexpectedly links CUDA.\n' >&2
    exit 1
fi

if ldd "${backend_target}" | grep -Eq 'libOpenFOAM|libfiniteVolume|libmeshTools|libcudart'; then
    printf 'ERROR: packaged backend dependency boundary check failed.\n' >&2
    exit 1
fi

printf 'Installed frontend: %s\n' "${frontend_target}"
printf 'Installed backend:  %s\n' "${backend_target}"
printf 'Installation check: PASS\n'
