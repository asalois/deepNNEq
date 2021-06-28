#!/bin/bash
# script to organize the output and make new dir
#docker run -it  -v /home/alexandersalois/Documents/tf_docker:/results tf /bin/bash
docker run -d  -v /home/alexandersalois/Documents/deepNNEq:/code tfgpu /bin/bash run_deep.sh 39

