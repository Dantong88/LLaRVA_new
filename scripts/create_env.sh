#!/bin/bash
set -e

# Get the directory of this script so that we can reference paths correctly no matter which folder
# the script was launched from:
SCRIPTS_DIR="$(builtin cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(realpath "${SCRIPTS_DIR}"/../)"

PYTHON_VERSION=3.10
ENV_NAME="llarva"
# * use export for this one so FORCE_CUDA is set for any sub-processes launched by this script:
export FORCE_CUDA=1
echo "ENV_NAME: ${ENV_NAME}"
echo "hostname: ${HOSTNAME}"

# * Load lmod Modules
set +e
echo ""
echo ""
echo "=========================================================="
echo "loading modules"
. $MODULESHOME/init/bash || true
if command -v module &>/dev/null; then
    # module load cuda || true # does not exist on warhawk
    echo "loading cseinit"
    module load cseinit || true
    module load cuda || true
    # nvcc -V || true

    if [[ $BC_HOST == "raider" ]]; then
        echo "loading modules for $BC_HOST"
    elif [[ $BC_HOST == "nautilus" ]]; then
        echo "loading modules for $BC_HOST"
    elif [[ $BC_HOST == "warhawk" ]]; then
        echo "loading modules for $BC_HOST"
        # error on raider (unable to locate a module file for nvidia/22.3)
        echo "loading nvidia/22.3"
        module load nvidia/22.3 || true
        nvcc -V || true
    else
        echo "Unknown center: $BC_HOST"
        exit 1
    fi
fi
set -e

echo ""
echo ""
echo "=========================================================="
echo "Loading conda..."
export PYTHONPATH=""
unset PYTHONPATH
echo "PYTHONPATH: ${PYTHONPATH}"
CONDA_FN="conda"
if [[ -d "${HOME}/mambaforge" ]]; then
    CONDA_FN="mamba"
    CONDA_DIR="${HOME}/mambaforge"
elif [[ -d "${HOME}/anaconda3" ]]; then
    CONDA_DIR="${HOME}/anaconda3"
elif [[ -d "${HOME}/miniconda3" ]]; then
    CONDA_DIR="${HOME}/miniconda3"
elif [[ -d "/global/common/software/nersc/pm-2022q3/sw/python/3.9-anaconda-2021.11" ]]; then
    ## If you use conda on an HPC cluster:
    CONDA_DIR="/global/common/software/nersc/pm-2022q3/sw/python/3.9-anaconda-2021.11"
fi

echo "CONDA_FN: $CONDA_FN"
echo "CONDA_DIR: $CONDA_DIR"

##
## Activate Conda (or Miniconda, or Mamba)
echo "Sourcing CONDA_FN: '$CONDA_FN' from location: '${CONDA_DIR}'"
if [ -d "${CONDA_DIR}/etc/profile.d" ]; then
    source "${CONDA_DIR}/etc/profile.d/conda.sh"
fi
if [ -f "${CONDA_DIR}/etc/profile.d/mamba.sh" ]; then
    source "${CONDA_DIR}/etc/profile.d/mamba.sh"
fi


##
## Remove env if exists:
set +e
echo ""
echo ""
echo "=========================================================="
CONDA_ENV_DIR=$(conda info | grep -i "envs directories" | sed "s/envs directories : //")
CONDA_ENV_DIR=$(echo "${CONDA_ENV_DIR}" | sed 's/[[:blank:]]//g')
CONDA_ENV_DIR="${CONDA_ENV_DIR}/${ENV_NAME}"
echo "Checking if we need to remove env ('${CONDA_ENV_DIR}')"
if [ -d "${CONDA_ENV_DIR}" ]; then
    echo "removing environment: ${ENV_NAME}"
    $CONDA_FN deactivate && $CONDA_FN env remove --name "${ENV_NAME}" -y || true
    echo "deleting ${CONDA_ENV_DIR}"
    rm -rf "${CONDA_ENV_DIR}" || true
fi
echo "Finished removing env"
ls -lah "${CONDA_ENV_DIR}" || true
set -e


##
## Create env:
echo ""
echo ""
echo "=========================================================="
echo "Creating conda env: ${ENV_NAME}"
$CONDA_FN create --name "${ENV_NAME}" python=="${PYTHON_VERSION}" -y
$CONDA_FN activate "${ENV_NAME}"
echo "Current environment: "
$CONDA_FN info --envs | grep "*"

$CONDA_FN install conda-libmamba-solver -y

##
## Base dependencies
echo "Installing requirements..."
function install_pytorch_cuda() {
    echo "Installing pytorch"
    $CONDA_FN install \
        pytorch[build=*cuda*,channel=pytorch,version=2.1.2] \
        pytorch-cuda==11.8 \
        torchaudio \
        torchvision \
        -c pytorch -c nvidia --solver=libmamba -y
}
install_pytorch_cuda
pip install --upgrade setuptools wheel

##
## Custom dependencies
## Move to project root
pushd "${PROJ_ROOT}"

## Install this repo (llarva):
pip install -e ".[train]"
pip install -r requirements_hpc.txt
pip install flash-attn --no-build-isolation --no-cache-dir

popd

# Some of the previous pip installs install their own torch/torchvision versions that might be
# incompatible with each other. Also, some of them replace the GPU version of pytorch with a CPU
# version. This final re-install of pytorch fixes it back:
echo ""
echo ""
echo "=========================================================="
echo "torch / torchvision fixup"

set +e
$CONDA_FN uninstall torch -y
$CONDA_FN uninstall pytorch -y
$CONDA_FN uninstall torchvision torchaudio -y
pip uninstall torch -y
pip uninstall pytorch -y
pip uninstall torchvision torchaudio -y
set -e

$CONDA_FN list
install_pytorch_cuda
## TODO: If script still doesn't work, try also re-installing flash-attn here:
# pip install -U flash-attn --no-build-isolation --no-cache-dir

## We are done, show the python environment:
$CONDA_FN list

## Check if we can load cuda:
"${PROJ_ROOT}/scripts/env_check.sh"
echo "Done!"
