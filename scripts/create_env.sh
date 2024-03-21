#!/bin/bash
set -e

# Get the directory of this script so that we can reference paths correctly no matter which folder
# the script was launched from:
SCRIPTS_DIR="$(builtin cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(realpath "${SCRIPTS_DIR}"/../)"

PYTHON_VERSION=3.10
ENV_NAME="llarva"
# use export for this one so FORCE_CUDA is set for any sub-processes launched by this script:
export FORCE_CUDA=1
echo "ENV_NAME: ${ENV_NAME}"

## Load lmod Modules
set +e
if command -v module &> /dev/null
then
    module load cuda
    # module load python/3.9-anaconda-2021.11
fi
set -e


# Feel free to customize this section, but this setup just picks the first comda or mamba
# installation tht it finds in the following order:
if [[ -d "${HOME}/mambaforge" ]]; then
    ## If you use mamba:
    CONDA_FN="mamba"
    CONDA_DIR="${HOME}/mambaforge"
elif [[ -d "${HOME}/anaconda3" ]]; then
    ## If you use conda on a normal (local) computer:
    CONDA_FN="conda"
    CONDA_DIR="${HOME}/anaconda3"
elif [[ -d "/global/common/software/nersc/pm-2022q3/sw/python/3.9-anaconda-2021.11" ]]; then
    ## If you use conda on an HPC cluster:
    CONDA_FN="conda"
    CONDA_DIR="/global/common/software/nersc/pm-2022q3/sw/python/3.9-anaconda-2021.11"
fi

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
if [ -d "${CONDA_DIR}/envs/${ENV_NAME}" ]; then
    $CONDA_FN deactivate && $CONDA_FN env remove --name "${ENV_NAME}"
    rm -rf "${CONDA_DIR}/envs/${ENV_NAME}"
fi
set -e

function install_pytorch_cuda() {
    echo "Installing pytorch"
    # $CONDA_FN install -y pytorch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0 cudatoolkit=11.3 -c pytorch -c conda-forge

    # If you have trouble installing torch, i.e, package manager installs cpu version, or wrong version,
    # might need to specify channel version, and cuda using this method. Additionally, it could help to
    # search the package repositories to see what packages and build versions are available, using:
    #   `conda search "pytorch[build=*cuda11.1*,version=1.8.1,channel=pytorch]"`
    #   `conda search "pytorch[build=*cuda12*,channel=pytorch]"`
    #
    # $CONDA_FN install \
    #     "pytorch[build=*cuda11.1*,version=1.8.1,channel=pytorch]" \
    #     "torchvision[build=*_cu111*,version=0.9.1,channel=pytorch]" \
    #     cudatoolkit=11.1 \
    #     -c pytorch -c conda-forge -c anaconda -y

    # $CONDA_FN install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -c conda-forge --solver=libmamba


    $CONDA_FN install \
        pytorch[build=*cuda*,channel=pytorch,version=2.1.2] \
        pytorch-cuda==11.8 \
        torchvision \
        -c pytorch -c nvidia -c conda-forge --solver=libmamba -y
}

##
## Create env:
$CONDA_FN create --name "${ENV_NAME}" python=="${PYTHON_VERSION}" -y
$CONDA_FN activate "${ENV_NAME}"
echo "Current environment: "
$CONDA_FN info --envs | grep "*"

$CONDA_FN install conda-libmamba-solver -y

##
## Base dependencies
echo "Installing requirements..."
install_pytorch_cuda
pip install --upgrade setuptools wheel

##
## Custom dependencies
## Move to project root
pushd "${PROJ_ROOT}"

## Install this repo (llarva):
pip install -e ".[train]"
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
echo "Doing a quick check for torch.cuda:"
python -c "import torch; print('torch.cuda.is_available: ', torch.cuda.is_available())"
echo "Done!"
