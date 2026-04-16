#!/usr/bin/env bash
# stage-cairo.sh — Populate cairo-stage/ from the Linux_for_Tegra aarch64 rootfs.
#
# WHY THIS EXISTS
# ---------------
# The Dockerfile COPYs cairo-stage/ into /usr/lib/aarch64-linux-gnu/ so that
# nvdsosd (DeepStream's bbox drawer) can dlopen libcairo.so.2 for Pango text
# rendering.  Installing libcairo2 via apt-get inside a buildx cross-build runs
# under qemu-binfmt aarch64 emulation, which takes 12+ minutes for two packages.
# Copying pre-staged .so files from the host rootfs finishes in seconds.
#
# WHEN TO RE-RUN
# --------------
# - After updating Linux_for_Tegra (L4T rootfs bump changes library versions).
# - After adding a DeepStream plugin whose GStreamer element dlopen-s a new lib
#   not already covered by this set (add the new root to SEEDS below).
# - After any manual edit to the file list that proves insufficient at runtime
#   (check docker logs for "error while loading shared libraries").
#
# USAGE
#   ./scripts/stage-cairo.sh [/path/to/Linux_for_Tegra/rootfs]
#
# Defaults to ~/jetson/Linux_for_Tegra/rootfs if no argument is given.

set -euo pipefail

ROOTFS="${1:-${HOME}/jetson/Linux_for_Tegra/rootfs}"
LIBDIR="${ROOTFS}/usr/lib/aarch64-linux-gnu"
STAGE_DIR="$(cd "$(dirname "$0")/.." && pwd)/cairo-stage"

# Libraries known to be present in the target container already (glibc, libz,
# libm, ld-linux) — stop the BFS traversal at these rather than staging them.
SKIP_PATTERN='^(libc\.so\.|libm\.so\.|libz\.so\.|ld-linux|libdl\.so\.|libpthread\.so\.|librt\.so\.)'

if [[ ! -d "${LIBDIR}" ]]; then
    echo "ERROR: rootfs lib dir not found: ${LIBDIR}" >&2
    echo "Pass the rootfs path as the first argument." >&2
    exit 1
fi

mkdir -p "${STAGE_DIR}"

# --------------------------------------------------------------------------
# Seed set: the direct dlopen targets.  readelf BFS derives the rest.
# Add new seeds here if a new DS plugin needs additional root libraries.
#
# libcairo.so.2      — the direct dlopen target of nvdsosd/Pango.
#
# libgraphite2.so.3  — NOT a direct NEEDED of libcairo.so.2, but required
#   at runtime because libharfbuzz (which IS a direct dep of cairo's Pango
#   text stack) dlopen-s it.  libharfbuzz itself is CDI-injected and already
#   present in the container; only its dependency libgraphite2 is missing from
#   the base ubuntu:24.04 image.  Adding it here as an explicit seed rather
#   than via BFS because libharfbuzz's other dep (libglib-2.0.so.0) is
#   CDI-injected and must NOT be over-staged from the rootfs.
# --------------------------------------------------------------------------
declare -a SEEDS=(
    libcairo.so.2
    libgraphite2.so.3
)

# --------------------------------------------------------------------------
# BFS over NEEDED entries.  For each soname we:
#   1. Resolve the soname symlink in LIBDIR to find the versioned filename.
#   2. Run readelf -d on the versioned file to extract NEEDED entries.
#   3. Enqueue any NEEDED entries not yet visited and not in SKIP_PATTERN.
#   4. Copy both the versioned file and its soname symlink to STAGE_DIR.
# --------------------------------------------------------------------------
declare -A visited

copy_lib() {
    local soname="$1"

    # Resolve soname -> versioned filename via the symlink in LIBDIR
    local src_link="${LIBDIR}/${soname}"
    if [[ ! -e "${src_link}" && ! -L "${src_link}" ]]; then
        echo "  SKIP  ${soname}  (not found in rootfs — likely already in container base)" >&2
        return
    fi

    # Follow the symlink one level to get the real filename
    local versioned
    versioned=$(readlink "${src_link}" 2>/dev/null || true)
    if [[ -z "${versioned}" ]]; then
        # src_link is itself the real file (no symlink), just copy it
        versioned="${soname}"
    fi
    # versioned may be a relative name like libcairo.so.2.11600.0
    local src_real="${LIBDIR}/${versioned}"

    # Copy the versioned file (preserve timestamps)
    if [[ ! -f "${STAGE_DIR}/${versioned}" ]]; then
        echo "  COPY  ${versioned}"
        cp -p "${src_real}" "${STAGE_DIR}/${versioned}"
    fi

    # Re-create the soname symlink (only if soname != versioned)
    if [[ "${soname}" != "${versioned}" && ! -L "${STAGE_DIR}/${soname}" ]]; then
        echo "  LINK  ${soname} -> ${versioned}"
        ln -sf "${versioned}" "${STAGE_DIR}/${soname}"
    fi
}

enqueue() {
    local soname="$1"
    if [[ -v "visited[${soname}]" ]]; then
        return
    fi
    visited["${soname}"]=1

    # Skip glibc and other libs that are always present in the base image
    if echo "${soname}" | grep -qE "${SKIP_PATTERN}"; then
        return
    fi

    copy_lib "${soname}"

    # Walk NEEDED entries of the versioned file to find transitive deps
    local src_link="${LIBDIR}/${soname}"
    if [[ ! -e "${src_link}" && ! -L "${src_link}" ]]; then
        return
    fi
    local versioned
    versioned=$(readlink "${src_link}" 2>/dev/null || true)
    if [[ -z "${versioned}" ]]; then
        versioned="${soname}"
    fi
    local src_real="${LIBDIR}/${versioned}"
    if [[ ! -f "${src_real}" ]]; then
        return
    fi

    while IFS= read -r needed; do
        enqueue "${needed}"
    done < <(readelf -d "${src_real}" 2>/dev/null \
              | grep '(NEEDED)' \
              | sed -n 's/.*Shared library: \[\(.*\)\]/\1/p')
}

echo "Staging cairo libs from: ${LIBDIR}"
echo "Destination: ${STAGE_DIR}"
echo ""

for seed in "${SEEDS[@]}"; do
    enqueue "${seed}"
done

echo ""
echo "Done. $(ls "${STAGE_DIR}" | wc -l) files in cairo-stage/."
echo ""
echo "If ldconfig reports missing libs at runtime, check the container logs for"
echo "'error while loading shared libraries' and add the missing soname to SEEDS."
