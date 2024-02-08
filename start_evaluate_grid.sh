#!/usr/bin/env bash
START=${START:-0.6}
STOP=${STOP:-1.0}
STEP=${STEP:-0.01}
EXPERIMENT_JSON=${EXPERIMENT_JSON:-"./experiments/s20003.json"}
for i in $(python -c "import numpy as np; [print(i) for i in np.arange(${START}, ${STOP}, ${STEP})]"); do
    cd /home/orb/repos/kinect_mocap_studio/
    echo ./build/evaluate -e ${EXPERIMENT_JSON} -r 0 -p 0 -f 1 -m $i -x 1
    ./build/evaluate -e ${EXPERIMENT_JSON} -r 0 -p 0 -f 1 -m $i -x 1
done
