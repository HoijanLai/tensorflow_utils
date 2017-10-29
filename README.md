### Usage
1. Download the bosch small traffic light dataset
2. Install tensorflow and the object detection API
3. In the script:
```
from object_detection.utils import dataset_util
```
that requires the script to run in `<installation_dir>/models/research`
4.run example:
```
python bosch_lights_to_tf_record.py \
--input_yaml=`pwd`/dataset_train_rgb/train.yaml \
--output_path=`pwd`/light_out.record
```