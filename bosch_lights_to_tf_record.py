import tensorflow as tf
from object_detection.utils import dataset_util
import yaml
import os
import cv2
from tqdm import tqdm

flags = tf.app.flags
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
FLAGS = flags.FLAGS


def get_all(input_yaml):
    """ Gets all labels within label file

    Note that RGB images are 1280x720
    :param input_yaml: Path to yaml file
    :param riib: If True, change path to labeled pictures
    :return: images: Labels for traffic lights
    """
    images = yaml.load(open(input_yaml, 'rb').read())
    for i in range(len(images)):
        images[i]['path'] = os.path.abspath(os.path.join(os.path.dirname(input_yaml), images[i]['path']))
    return images





def create_tf_example(example):
    # TODO(user): Populate the following variables from your example.
    height = 720 # Image height
    width = 1280 # Image width
    filename = str.encode(example['path']) # Filename of the image. Empty if image is not from file
    im = cv2.imread(example['path'])
    encoded_image_data = im.tobytes() # Encoded image bytes
    image_format = b'png' # b'jpeg' or b'png'
    
    boxes = example['boxes']
    
    
    xmins = [] 
    xmaxs = [] 
             
    ymins = [] 
    ymaxs = [] 
             
    classes_text = [] 
    classes = [] 

    for box in boxes:
        bw = box['x_max'] - box['x_min']
        bh = box['y_max'] - box['y_min']
        
        if bw < 0 or box['label'] == 'off':
            continue
        
        xmins.append(box['x_min']/width)
        xmaxs.append(box['x_max']/width)
        ymins.append(box['y_min']/height)
        ymaxs.append(box['y_max']/height)
        text = box['label'] 
        if 'Green' in text:
            classes_text.append(b'Green')
            classes.append(2)
        elif 'Red' in text:
            classes_text.append(b'Red')
            classes.append(0)
        elif 'Yellow' in text:
            classes_text.append(b'Yellow')
            classes.append(1)
            

        
    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_image_data),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example




def main(_):
    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)
    path = './dataset_additional_rgb/additional_train.yaml'
    examples = get_all(path)

    for example in tqdm(examples):
        tf_example = create_tf_example(example)
        if tf_example:
            writer.write(tf_example.SerializeToString())

    writer.close()



if __name__ == '__main__':
    tf.app.run()
