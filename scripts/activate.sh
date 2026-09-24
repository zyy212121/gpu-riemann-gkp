#!/usr/bin/env bash
_gpu_gkp_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${_gpu_gkp_root}/scripts/openfoam10-env.sh" || return 1
export FOAM_USER_APPBIN="${_gpu_gkp_root}/build/bin"
export PATH="${_gpu_gkp_root}/scripts/bin:${FOAM_USER_APPBIN}:${PATH}"
export GAS_UGKP_CUDA_BACKEND="${FOAM_USER_APPBIN}/gasUGKPCudaBackend"
unset _gpu_gkp_root
