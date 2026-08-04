#!/usr/bin/env bash

set -euo pipefail

case_dir="$(readlink -f "$1")"
shift
resolved_case="$(readlink -f "${case_dir}")"

if [[ "${resolved_case}/" != *"/cp_cst_test/"* ]]; then
    echo "ERROR: refusing to clean outside cp_cst_test: ${resolved_case}" >&2
    exit 1
fi

while IFS= read -r time_dir
do
    time_name="$(basename "${time_dir}")"
    if awk -v value="${time_name}" 'BEGIN { exit !(value > 0.100000021) }'
    then
        rm -rf -- "${time_dir}"
    fi
done < <(
    find "${case_dir}" -mindepth 1 -maxdepth 1 -type d \
        -regextype posix-extended \
        -regex '.*/[0-9]+(\.[0-9]+)?' \
        -print
)

rm -f -- "${case_dir}/log.gpu" "${case_dir}/runtime.gpu"
