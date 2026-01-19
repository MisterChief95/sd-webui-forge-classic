@echo off

:: set PYTHON=
:: set GIT=
:: set VENV_DIR=
set DEBUG_MEMORY=1
set COMMANDLINE_ARGS=--adv-samplers --uv --skip-python-version-check --skip-torch-cuda-test --skip-version-check --skip-prepare-environment --skip-install --cuda-malloc --cuda-stream
:: --xformers --sage --uv
:: --pin-shared-memory --cuda-malloc --cuda-stream
:: --skip-python-version-check --skip-torch-cuda-test --skip-version-check --skip-prepare-environment --skip-install

call webui.bat
