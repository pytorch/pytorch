# Reporting Security Issues

If you believe you have found a security vulnerability in PyTorch, we encourage you to let us know right away. We will investigate all legitimate reports and do our best to quickly fix the problem.

Please report security issues using https://github.com/pytorch/pytorch/security/advisories/new

Please refer to the following page for our responsible disclosure policy, reward guidelines, and those things that should not be reported:

https://www.facebook.com/whitehat

# conda install -c main pytorch

# Código
Asuntos
5k+
Más
Inyección de expresión de acciones en `filter-test-configs` (`GHSL-2023-181`)
Moderado	malfet publicado GHSA-hw6r-g8gj-2987 la semana pasada
Paquete
 pytorch/pytorch/.github/actions/filter-test-configs 
(
Acciones de GitHub
)
Versiones afectadas
<v2.0.1
Versiones parcheadas
Ninguno
Descripción
El pytorch/pytorch filter-test-configsflujo de trabajo es vulnerable a una inyección de expresión en Acciones, lo que permite a un atacante potencialmente filtrar secretos y alterar el repositorio utilizando el flujo de trabajo.

Detalles
El filter-test-configsflujo de trabajo utiliza el github.event.workflow_run.head_branchvalor bruto dentro del filterpaso:

- name: Select all requested test configurations
  shell: bash
  env:
    GITHUB_TOKEN: ${{ inputs.github-token }}
    JOB_NAME: ${{ steps.get-job-name.outputs.job-name }}
  id: filter
  run: |
    ...
    python3 "${GITHUB_ACTION_PATH}/../../scripts/filter_test_configs.py" \
      ...
      --branch "${{ github.event.workflow_run.head_branch }}"
En el caso de que un repositorio se utilice filter-test-configsen un pull_request_targetflujo de trabajo activado, un atacante podría utilizar un nombre de rama malicioso para obtener la ejecución de comandos en el paso y potencialmente filtrar secretos.

name: Example

on: pull_request_target

jobs:
  example:
    runs-on: ubuntu-latest
    steps:
      - name: Filter
        uses: pytorch/pytorch/.github/actions/filter-test-configs@v2
Impacto
Este problema puede provocar el robo de secretos del flujo de trabajo.

Remediación
Utilice una variable de entorno intermedia para valores potencialmente controlados por atacantes como github.event.workflow_run.head_branch:
- name: Select all requested test configurations
  shell: bash
  env:
    GITHUB_TOKEN: ${{ inputs.github-token }}
    JOB_NAME: ${{ steps.get-job-name.outputs.job-name }}
    HEAD_BRANCH: ${{ github.event.workflow_run.head_branch }}
  id: filter
  run: |
    ...
    python3 "${GITHUB_ACTION_PATH}/../../scripts/filter_test_configs.py" \
      ...
      --branch "$HEAD_BRANCH"

- name: Select all requested test configurations
  shell: bash
  env:
    GITHUB_TOKEN: ${{ inputs.github-token }}
    JOB_NAME: ${{ steps.get-job-name.outputs.job-name }}
    HEAD_BRANCH: ${{ github.event.workflow_run.head_branch }}
  id: filter
  run: |
    ...
    python3 "${GITHUB_ACTION_PATH}/../../scripts/filter_test_configs.py" \
      ...
      --branch "$HEAD_BRANCH"
