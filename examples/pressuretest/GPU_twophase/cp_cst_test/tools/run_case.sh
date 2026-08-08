#!/usr/bin/env bash

set -o pipefail

case_dir="$(readlink -f "$1")"
shift
level="$(basename "${case_dir}")"
parent_name="$(basename "$(dirname "${case_dir}")")"

if [[ "${parent_name}" == "dense" ]]; then
    suite_dir="$(cd "${case_dir}/../.." && pwd)"
    group="dense"
    expected_mass="1e-11"
    expected_capacity="8000000"
else
    suite_dir="$(cd "${case_dir}/.." && pwd)"
    group="sparse"
    expected_mass="5e-09"
    expected_capacity="1000000"
fi

solver_root="$(cd "${suite_dir}/../../../.." && pwd)"

source "$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)/scripts/openfoam10-env.sh"
set -eu

user_foam_root="$(cd "${solver_root}/../../.." && pwd)"
export FOAM_USER_APPBIN="${FOAM_USER_APPBIN:-${user_foam_root}/platforms/${WM_OPTIONS}/bin}"
frontend="${FOAM_USER_APPBIN}/GpuGkp"
backend="${FOAM_USER_APPBIN}/gpu28CudaBackend"

case "${level}" in
    l0)
        expected_path="false"
        expected_heavy="false"
        expected_warp="false"
        ;;
    l1)
        expected_path="true"
        expected_heavy="false"
        expected_warp="false"
        ;;
    e1)
        expected_path="true"
        expected_heavy="false"
        expected_warp="true"
        ;;
    e2)
        expected_path="true"
        expected_heavy="true"
        expected_warp="true"
        ;;
    *)
        echo "ERROR: unsupported CP-CST test level: ${level}" >&2
        exit 1
        ;;
esac

dictionary_value()
{
    foamDictionary "$1" -entry "$2" -value | tr -d '[:space:];'
}

require_value()
{
    local file="$1"
    local entry="$2"
    local expected="$3"
    local actual
    actual="$(dictionary_value "${file}" "${entry}")"
    if [[ "${actual}" != "${expected}" ]]; then
        echo "ERROR: ${file}:${entry}=${actual}, expected ${expected}" >&2
        exit 1
    fi
}

if pgrep -f '[d]iluteUgkwpFoam_GPU2_8|[g]pu26CudaBackend' >/dev/null; then
    echo "ERROR: another GPU2.8 calculation is already running." >&2
    exit 1
fi

test -x "${frontend}"
test -x "${backend}"
test -d "${case_dir}/0.10000002"

cd "${case_dir}"

if [[ ! -f constant/polyMesh/owner ]]; then
    blockMesh > log.blockMesh 2>&1
fi

require_value system/controlDict startFrom latestTime
require_value system/controlDict endTime 0.11
require_value system/controlDict writeInterval 0.01
require_value constant/ugkwpProperties gpuResidentPureGasOnly false
require_value constant/ugkwpProperties particleTemperatureTransport false
require_value constant/ugkwpProperties parcelMass "${expected_mass}"
require_value constant/ugkwpProperties gpuResidentParticleCapacity "${expected_capacity}"
require_value constant/ugkwpProperties gpuCsrCellLocalPath "${expected_path}"
require_value constant/ugkwpProperties gpuCsrHeavyReduction "${expected_heavy}"
require_value constant/ugkwpProperties gpuCsrHeavyCellThreshold 1024
require_value constant/ugkwpProperties gpuCsrHeavyTileParticles 1024
require_value constant/ugkwpProperties gpuCsrHeavyWorkerBlocksPerSM 4
require_value constant/ugkwpProperties gpuCsrWarpAggregatedBinning "${expected_warp}"

if [[ "${CP_CST_CHECK_ONLY:-0}" == "1" ]]; then
    echo "Configuration check passed for ${group}/${level}."
    exit 0
fi

if [[ -e log.gpu || -e runtime.gpu ]]; then
    echo "ERROR: refusing to overwrite ${case_dir}/log.gpu or runtime.gpu" >&2
    exit 1
fi

export GPU28_CUDA_BACKEND="${backend}"

echo "Starting ${group}/${level}: runtime CP-CST path=${expected_path}" >&2
exec /usr/bin/time \
    -f 'elapsed_seconds=%e' \
    -o runtime.gpu \
    "${frontend}" \
    > log.gpu 2>&1
