#!/bin/bash
set -euo pipefail

# Interactive sediment MMS run on Perlmutter CPU nodes.
# First get an allocation, for example:
#   salloc -A m3780 -C cpu -q interactive -t 00:15:00 -N 1

RDYCORE_DIR=${RDYCORE_DIR:-/global/cfs/projectdirs/m4267/dongyu/rdycore_hr_sed}
MACH=${MACH:-pm-cpu}
CONFIG=${CONFIG:-1}
NP=${NP:-2}
YAML_FILE=${YAML_FILE:-sediment_mms_conv_study_3class.yaml}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source "${RDYCORE_DIR}/config/set_petsc_settings.sh" --mach "${MACH}" --config "${CONFIG}"

BUILD="${RDYCORE_DIR}/build-${PETSC_ARCH}"
RDYCORE_MMS="${BUILD}/driver/rdycore_mms"
MESH_FILE="${SCRIPT_DIR}/mms_triangles_dx1.exo"
MESH_SOURCE="${RDYCORE_DIR}/share/meshes/mms_triangles_dx1.exo"

if [[ ! -x "${RDYCORE_MMS}" ]]; then
  echo "ERROR: rdycore_mms executable not found or not executable:"
  echo "  ${RDYCORE_MMS}"
  exit 1
fi

if [[ ! -f "${SCRIPT_DIR}/${YAML_FILE}" ]]; then
  echo "ERROR: YAML file not found:"
  echo "  ${SCRIPT_DIR}/${YAML_FILE}"
  exit 1
fi

if [[ ! -e "${MESH_FILE}" ]]; then
  ln -s "${MESH_SOURCE}" "${MESH_FILE}"
fi

cd "${SCRIPT_DIR}"

echo "Running sediment MMS"
echo "  directory : ${SCRIPT_DIR}"
echo "  executable: ${RDYCORE_MMS}"
echo "  YAML      : ${YAML_FILE}"
echo "  MPI ranks : ${NP}"
echo

srun -n "${NP}" "${RDYCORE_MMS}" "${YAML_FILE}"
