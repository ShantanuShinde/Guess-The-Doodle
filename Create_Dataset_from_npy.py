import tensorflow as tf
import numpy as np
import os
import random
import argparse
import sys







def preprocessing(drawing):
    """
    Args:
        numpy array.
    Returns:
        numpy array that is normalized and deltas computed
    """
    #normalization
    lower = np.min(drawing[:,0:2],axis=0)
    higher = np.max(drawing[:,0:2],axis=0)  
    scale = higher - lower
    scale[scale == 0] = 1
    drawing[:,0:2] = (drawing[:,0:2] - lower)/scale

    #compute deltas
    drawing[1:,0:2] -= drawing[0:-1,0:2]
    drawing = drawing[1:,:]

    return drawing

def convert_to_tfrecord(source_path, destination_path, train_examples_per_class,eval_examples_per_class, classes_path,output_shards=10):
    """Convert numpy array from .npy files into tfrecords
    Args:
        source_path: Path of the .npy files
        destination_path: Path to store the tfrecords
        classes_path: Path to the file with list of names of classes
        training_examples_per_class: No. of drawings in each class for training
        eval_examples_per_class: No. of drawings in each class for evaluation
        output_shards: no. of shards to write output in
    
    Returns:
        list of class names
    """
    print("convert_to_tfrecord entered")
    with open(classes_path) as label_class:
        classes = label_class.readlines()
    classes = [x.strip() for x in classes]
    print("classes list created")

    def pick_output_shard():
        return random.randint(0, output_shards - 1)

    file_list = list(os.scandir(source_path))

    train_writers = []
    train_file = str(os.path.join(destination_path,"training.tfrecord"))
    for i in range(output_shards):
        train_writers.append(tf.python_io.TFRecordWriter(train_file + "-" + str(i) + "-of-" + str(output_shards)))
    print("Train writers created")
    eval_writers = []
    eval_file = str(os.path.join(destination_path,"eval.tfrecord"))
    for i in range(output_shards):
        eval_writers.append(tf.python_io.TFRecordWriter(eval_file + "-" + str(i) + "-of-" + str(output_shards)))
    print("Eval writers created")
    
    
    
    file_ind = list(range(len(file_list)))
    train_ind = file_ind*train_examples_per_class
    eval_ind = file_ind*eval_examples_per_class
    random.shuffle(train_ind)
    random.shuffle(eval_ind)
    #read = [[] for x in file_ind]
    ind_to_read = [0]*len(file_ind)
    
    #Save training examples
    for fl_ind in train_ind:    
        drawings = np.load(file_list[fl_ind].path)
        drawing = drawings[ind_to_read[fl_ind]]
        drawing = np.reshape(drawing,(28,28))
        ind_to_read[fl_ind]+=1
        drawing = preprocessing(drawing)
        print(file_list[fl_ind],ind_to_read[fl_ind])
        features = {}
        features["class_index"] = tf.train.Feature(int64_list = tf.train.Int64List(value=[classes.index(file_list[fl_ind].name[18:-4])]))
        features["drawing"] = tf.train.Feature(float_list=tf.train.FloatList(value=drawing.flatten()))
        features["shape"] = tf.train.Feature(int64_list = tf.train.Int64List(value=drawing.shape))
        ftr = tf.train.Features(feature=features)
        example = tf.train.Example(features=ftr)
        train_writers[pick_output_shard()].write(example.SerializeToString())
       
    for w in train_writers:
            w.close()

    #Save evaluation examples
    for fl_ind in eval_ind:    
        drawings = np.load(file_list[fl_ind].path)
        drawing = drawings[ind_to_read[fl_ind]]
        drawing = np.reshape(drawing,(28,28))
        ind_to_read[fl_ind]+=1
        drawing = preprocessing(drawing)
        print(file_list[fl_ind],ind_to_read[fl_ind])
        features = {}
        features["class_index"] = tf.train.Feature(int64_list = tf.train.Int64List(value=[classes.index(file_list[fl_ind].name[18:-4])]))
        features["drawing"] = tf.train.Feature(float_list=tf.train.FloatList(value=drawing.flatten()))
        features["shape"] = tf.train.Feature(int64_list = tf.train.Int64List(value=drawing.shape))
        ftr = tf.train.Features(feature=features)
        example = tf.train.Example(features=ftr)
        eval_writers[pick_output_shard()].write(example.SerializeToString())

    for w in eval_writers:
            w.close()

    
    
        
    return classes
 
def main(argv):
    print("Main entered")
    del argv
    classes = convert_to_tfrecord(FLAGS.source_path,FLAGS.destination_path,FLAGS.train_examples_per_class,FLAGS.eval_examples_per_class,FLAGS.classes_path,FLAGS.output_shards)
    
if __name__ == "__main__":
    print("Program started")
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true") 
    parser.add_argument(
        "--source_path",
        type=str,
        default="dataset",
        help="Directory where npy files are stored"
    )
    parser.add_argument(
        "--destination_path",
        type=str,
        default="TFRecord dataset",
        help="Directory to store tfrecord files"
    )
    parser.add_argument(
        "--classes_path",
        type=str,
        default="quickdraw-dataset-master\categories.txt",
        help="File with list of classes"
    )
    parser.add_argument(
        "--train_examples_per_class",
        type=float,
        default=10000,
        help="No. of examples for training, in each class"
    )
    parser.add_argument(
        "--eval_examples_per_class",
        type=float,
        default=1000,
        help="No. of examples for evaluation, in each class"
    )
    parser.add_argument(
        "--output_shards",
        type=int,
        default=10,
        help="No. of shards for output"
    )
    print("Arguments added")
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main,argv=[sys.argv[0]]+unparsed)
    






        

        
        



    

    
