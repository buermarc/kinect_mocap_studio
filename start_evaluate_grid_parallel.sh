#!/usr/bin/env bash
START=${START:-0}
STOP=${STOP:-30.02}
STEP=${STEP:-0.1}
EXPERIMENT_JSON=${EXPERIMENT_JSON:-"./experiments/s20003.json"}
python -c "import numpy as np; [print(i) for i in np.arange(${START}, ${STOP}, ${STEP})]" > /tmp/args
cd /home/orb/repos/kinect_mocap_studio/
cat /tmp/args
cat /tmp/args | parallel -j12 ./build/evaluate -e ${EXPERIMENT_JSON} -r 0 -p 0 -f 1 -m {} -x 1
