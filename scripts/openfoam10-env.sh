#!/usr/bin/env bash

# This file is sourced by install.sh and the example runners.
if [[ "${WM_PROJECT_VERSION:-}" == "10" && -n "${FOAM_USER_APPBIN:-}" ]]; then
    return 0 2>/dev/null || exit 0
fi

openfoam_bashrc="${OPENFOAM_BASHRC:-}"
if [[ -z "${openfoam_bashrc}" ]]; then
    for candidate in \
        /opt/openfoam10/etc/bashrc \
        /usr/lib/openfoam/openfoam10/etc/bashrc
    do
        if [[ -f "${candidate}" ]]; then
            openfoam_bashrc="${candidate}"
            break
        fi
    done
fi

if [[ -z "${openfoam_bashrc}" || ! -f "${openfoam_bashrc}" ]]; then
    printf '%s\n' \
        'ERROR: OpenFOAM 10 environment was not found.' \
        'Source OpenFOAM 10 before running this command, or set:' \
        '  export OPENFOAM_BASHRC=/absolute/path/to/openfoam10/etc/bashrc' >&2
    return 1 2>/dev/null || exit 1
fi

# OpenFOAM 10 checks a few optional shell variables while loading. Preserve the
# caller's nounset mode and suspend it only for the environment import.
nounset_was_enabled=false
errexit_was_enabled=false
case "$-" in
    *u*) nounset_was_enabled=true; set +u ;;
esac
case "$-" in
    *e*) errexit_was_enabled=true; set +e ;;
esac

# OpenFOAM bashrc is intentionally sourced after resolving the path.
# shellcheck disable=SC1090
source "${openfoam_bashrc}"

if [[ "${errexit_was_enabled}" == true ]]; then
    set -e
fi
if [[ "${nounset_was_enabled}" == true ]]; then
    set -u
fi

if [[ "${WM_PROJECT_VERSION:-}" != "10" || -z "${FOAM_USER_APPBIN:-}" ]]; then
    printf 'ERROR: expected OpenFOAM 10, found WM_PROJECT_VERSION=%s\n' \
        "${WM_PROJECT_VERSION:-unset}" >&2
    return 1 2>/dev/null || exit 1
fi

unset openfoam_bashrc candidate nounset_was_enabled errexit_was_enabled
