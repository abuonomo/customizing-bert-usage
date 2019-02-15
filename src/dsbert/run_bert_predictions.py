import argparse
import logging
import sys
import os
import re
from pathlib import Path
import yaml

sys.path.append('bert')

from run_classifier import *

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)


def find_checkpoint(indir):
    files = os.listdir(indir)
    r = re.compile('model.ckpt-[0-9]+.index')
    checkpoints = list(filter(r.match, files))
    values = [int(re.search(r'\d+', checkpoint).group()) for checkpoint in checkpoints]
    i = values.index(max(values))
    ckpt = checkpoints[i]
    init_checkpoint = '{}/{}'.format(indir, ckpt)
    LOG.info('Found checkpoint at \"{}\"'.format(init_checkpoint))
    return init_checkpoint


def get_lines(test_str, text_file):
    if (test_str is not None) and (text_file is not None):
        LOG.error('Cannot pass both text_str and text_file args.')
        exit()
    elif (test_str is not None):
        lines = [(), ('0', test_str)]
    elif (text_file is not None):
        with open(text_file, 'r') as f0:
            texts = f0.readlines()
        LOG.error('Loaded {} strings from {}.'.format(len(texts), text_file))
        lines = [('0', txt) for txt in texts]
        lines.insert(0, ())
    else:
        LOG.error("No text supplied")
        exit()
    return lines


def main(in_config, test_str=None, text_file=None):
    lines = get_lines(test_str, text_file)

    LOG.info('Loading configuration from {}'.format(in_config))
    with open(in_config, 'r') as f0:
        config = config_parse(yaml.load(f0))

    bert_config_file = config['paths']['bert_config_file']
    bert_config = modeling.BertConfig.from_json_file(bert_config_file)
    output_dir = config['paths']['output_dir']
    learning_rate = config['parameters']['learning_rate']
    num_train_steps = config['parameters']['num_train_steps']
    num_warmup_steps = config['parameters']['num_warmup_steps']
    train_batch_size = config['parameters']['train_batch_size']
    eval_batch_size = config['parameters']['eval_batch_size']
    predict_batch_size = config['parameters']['predict_batch_size']
    save_checkpoints_steps = config['parameters']['save_checkpoints_steps']
    iterations_per_loop = config['parameters']['iterations_per_loop']
    num_tpu_cores = config['parameters']['num_tpu_cores']
    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    processor = STIProcessor()
    label_list = processor.get_labels()
    vocab_file = config['paths']['vocab_file']
    do_lower_case = config['general']['do_lower_case']
    term = config['general']['term']
    indir = config['paths']['indir']
    model_dir = indir
    init_checkpoint = find_checkpoint(indir)

    predictions = {}

    tpu_cluster_resolver = None
    master = None

    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        master=master,
        model_dir=model_dir,
        save_checkpoints_steps=save_checkpoints_steps,
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=iterations_per_loop,
            num_shards=num_tpu_cores,
            per_host_input_for_training=is_per_host))

    model_fn = model_fn_builder(
        bert_config=bert_config,
        num_labels=len(label_list),
        init_checkpoint=init_checkpoint,
        learning_rate=learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_tpu=False,
        use_one_hot_embeddings=False)

    # If TPU is not available, this will fall back to normal Estimator on CPU
    # or GPU.
    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=False,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        predict_batch_size=predict_batch_size)

    tokenizer = tokenization.FullTokenizer(
        vocab_file=vocab_file, do_lower_case=do_lower_case)

    predict_examples = processor._create_examples(lines, 'test')
    predict_file = os.path.join(output_dir, "predict.tf_record")

    max_seq_length = 128
    label_list = ["0", "1"]
    file_based_convert_examples_to_features(predict_examples, label_list,
                                            max_seq_length, tokenizer,
                                            predict_file)

    predict_batch_size = 8
    tf.logging.info("***** Running prediction*****")
    tf.logging.info("  Num examples = %d", len(predict_examples))
    tf.logging.info("  Batch size = %d", predict_batch_size)

    predict_drop_remainder = False
    predict_input_fn = file_based_input_fn_builder(
        input_file=predict_file,
        seq_length=max_seq_length,
        is_training=False,
        drop_remainder=predict_drop_remainder)
    import pdb; pdb.set_trace()
    result = estimator.predict(input_fn=predict_input_fn)

    output_predict_file = os.path.join(output_dir, "test_results.tsv")
    predictions[term] = []
    with tf.gfile.GFile(output_predict_file, "w") as writer:
        tf.logging.info("***** Predict results *****")
        results = list(result)
        for prediction in results:
            output_line = "\t".join(str(class_probability) for class_probability in prediction) + "\n"
            writer.write(output_line)
            LOG.info(prediction)
            predictions[term].append(prediction)

    LOG.info('Result: {}'.format(predictions))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="what does this script do?")
    parser.add_argument('in_config', help='yaml configuration file')
    parser.add_argument('--text', help='text on which to make prediction', type=str)
    parser.add_argument('--text_file', help='text on which to make prediction', type=str)
    args = parser.parse_args()

    main(args.in_config, args.text, args.text_file)