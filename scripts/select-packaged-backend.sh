#!/usr/bin/env bash
set -euo pipefail

usage()
{
    printf 'Usage: %s [--check] production|twophaseflux-validation [status-directory]\n' \
        "${0##*/}" >&2
}

check_only=false
if [[ "${1:-}" == "--check" ]]; then
    check_only=true
    shift
fi

[[ $# -ge 1 && $# -le 2 ]] || { usage; exit 2; }
variant="$1"
status_directory="${2:-}"

repo_root="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
# shellcheck disable=SC1091
source "${repo_root}/scripts/openfoam10-env.sh"

case "${variant}" in
    production)
        packaged_name="gpu28CudaBackend"
        display_name="production"
        ;;
    twophaseflux-validation)
        packaged_name="gpu28CudaBackend-twophaseflux-validation"
        display_name="two-phase flux validation"
        ;;
    *)
        usage
        exit 2
        ;;
esac

packaged_backend="${repo_root}/backend/linux-x86_64/${packaged_name}"
installed_backend="${FOAM_USER_APPBIN}/gpu28CudaBackend"

(
    cd "${repo_root}/backend"
    sha256sum --check --status SHA256SUMS
)
test -f "${packaged_backend}"

if ldd "${packaged_backend}" | grep -Eq 'libOpenFOAM|libfiniteVolume|libmeshTools|libcudart'; then
    printf 'ERROR: packaged backend dependency boundary check failed.\n' >&2
    exit 1
fi

if [[ "${check_only}" == true ]]; then
    printf 'Packaged %s backend check: PASS\n' "${display_name}"
    exit 0
fi

if pgrep -f '(^|/)GpuGkp([[:space:]]|$)|(^|/)gpu28CudaBackend([[:space:]]|$)' >/dev/null; then
    printf 'ERROR: a GPU-Riemann-GKP process is active; backend switch refused.\n' >&2
    exit 1
fi

mkdir -p "${FOAM_USER_APPBIN}"
install -m 0755 "${packaged_backend}" "${installed_backend}"

if [[ -n "${status_directory}" ]]; then
    mkdir -p "${status_directory}"
    {
        printf '# Active GPU backend\n\n'
        printf -- '- Mode: `%s`\n' "${display_name}"
        printf -- '- SHA-256: `%s`\n' "$(sha256sum "${installed_backend}" | awk '{print $1}')"
    } > "${status_directory}/ACTIVE_GPU28_BINARY.md"
fi

printf 'Installed %s backend: %s\n' "${display_name}" "${installed_backend}"
