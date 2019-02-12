import argparse
import logging
import os
import re

import yaml
import tokenization
import modeling as modeling
import optimization as optimization
from run_classifier import DataProcessor, InputExample, model_fn_builder, file_based_convert_examples_to_features, file_based_input_fn_builder
import tensorflow as tf
import numpy as np

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)


class STIProcessor(DataProcessor):
    """Processor for the NASA STI data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        # return ["0", "1"]
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0 or len(line) < 2:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = tokenization.convert_to_unicode(line[1])
            if set_type == "test":
                label = "0"
            else:
                label = tokenization.convert_to_unicode(line[0])
            examples.append(
                InputExample(guid=guid, text_a=text_a, label=label))
        return examples


def find_checkpoint(indir):
    LOG.info('Finding checkpoint in {}'.format(indir))
    files = os.listdir(indir)
    r = re.compile('model.ckpt-[0-9]+.index')
    checkpoints = list(filter(r.match, files))
    values = [int(re.search(r'\d+', checkpoint).group()) for checkpoint in checkpoints]
    i = values.index(max(values))
    ckpt = checkpoints[i]
    init_checkpoint = '{}/{}'.format(indir, ckpt)
    LOG.info('Found checkpoint at \"{}\"'.format(init_checkpoint))
    return init_checkpoint


def parse_config_nones(config):
    for k0, v0 in config.items():
        for k1, v1 in v0.items():
            if v1 == "None":
                v0[k1] = None
            else:
                continue
    return config


def bert_main(data_dir, bert_config_file, vocab_file, output_dir, init_checkpoint=None,
              do_train=False, do_eval=False, do_predict=False, do_lower_case=False, text_list=None,
              task_name='sti', save_checkpoints_steps=1000, iterations_per_loop=1000,
              train_batch_size=32, eval_batch_size=32, predict_batch_size=8, max_seq_length=128,
              num_train_epochs=3.0, warmup_proportion=0.1, learning_rate=2e-5, num_tpu_cores=8):
    tf.logging.set_verbosity(tf.logging.INFO)

    processors = {"sti": STIProcessor}
    if not do_train and not do_eval and not do_predict:
        raise ValueError(
            "At least one of `do_train`, `do_eval` or `do_predict' must be True.")

    bert_config = modeling.BertConfig.from_json_file(bert_config_file)
    if max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (max_seq_length, bert_config.max_position_embeddings))

    tf.gfile.MakeDirs(output_dir)
    task_name = task_name.lower()
    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()
    label_list = processor.get_labels()
    tokenizer = tokenization.FullTokenizer(
        vocab_file=vocab_file, do_lower_case=do_lower_case)

    if do_predict:
        model_dir = data_dir
    else:
        model_dir = output_dir
    tpu_cluster_resolver = None
    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        master=None,
        model_dir=model_dir,
        save_checkpoints_steps=save_checkpoints_steps,
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=iterations_per_loop,
            num_shards=num_tpu_cores,
            per_host_input_for_training=is_per_host))

    train_examples = None
    num_train_steps = None
    num_warmup_steps = None
    if do_train:
        train_examples = processor.get_train_examples(data_dir)
        num_train_steps = int(
            len(train_examples) / train_batch_size * num_train_epochs)
        assert num_train_steps != 0, LOG.error('Not enough data. num_train_steps is 0.')
        num_warmup_steps = int(num_train_steps * warmup_proportion)

    if do_predict:
        init_checkpoint = find_checkpoint(data_dir)

    model_fn = model_fn_builder(
        bert_config=bert_config,
        num_labels=len(label_list),
        init_checkpoint=init_checkpoint,
        learning_rate=learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_tpu=False,
        use_one_hot_embeddings=False)

    # If TPU is not available, this will fall back to normal Estimator on CPU or GPU.
    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=False,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        predict_batch_size=predict_batch_size)

    if do_train:
        train_file = os.path.join(output_dir, "train.tf_record")
        file_based_convert_examples_to_features(
            train_examples, label_list, max_seq_length, tokenizer, train_file)
        tf.logging.info("***** Running training *****")
        tf.logging.info("  Num examples = %d", len(train_examples))
        tf.logging.info("  Batch size = %d", train_batch_size)
        tf.logging.info("  Num steps = %d", num_train_steps)
        train_input_fn = file_based_input_fn_builder(
            input_file=train_file,
            seq_length=max_seq_length,
            is_training=True,
            drop_remainder=True)
        LOG.info('TRAIN STEPS: {}'.format(num_train_steps))
        estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

    if do_eval:
        LOG.info('EVAL! data dir: {}'.format(data_dir))
        eval_examples = processor.get_dev_examples(data_dir)
        eval_file = os.path.join(output_dir, "eval.tf_record")
        file_based_convert_examples_to_features(
            eval_examples, label_list, max_seq_length, tokenizer, eval_file)

        tf.logging.info("***** Running evaluation *****")
        tf.logging.info("  Num examples = %d", len(eval_examples))
        tf.logging.info("  Batch size = %d", eval_batch_size)

        # This tells the estimator to run through the entire set.
        eval_steps = None
        eval_drop_remainder = False
        eval_input_fn = file_based_input_fn_builder(
            input_file=eval_file,
            seq_length=max_seq_length,
            is_training=False,
            drop_remainder=eval_drop_remainder)

        result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)

        output_eval_file = os.path.join(output_dir, "eval_results.txt")
        with tf.gfile.GFile(output_eval_file, "w") as writer:
            tf.logging.info("***** Eval results *****")
            for key in sorted(result.keys()):
                tf.logging.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

    if do_predict:
        LOG.info('PREDICT! data dir: {}'.format(data_dir))
        if text_list is not None: # pass list directly within python
            lines = [('0', txt) for txt in text_list]
            lines.insert(0, ())
            predict_examples = processor._create_examples(lines, 'test')
        else:
            predict_examples = processor.get_test_examples(data_dir)
        predict_file = os.path.join(output_dir, "predict.tf_record")
        file_based_convert_examples_to_features(predict_examples, label_list,
                                                max_seq_length, tokenizer,
                                                predict_file)

        tf.logging.info("***** Running prediction*****")
        tf.logging.info("  Num examples = %d", len(predict_examples))
        tf.logging.info("  Batch size = %d", predict_batch_size)

        predict_drop_remainder = False
        predict_input_fn = file_based_input_fn_builder(
            input_file=predict_file,
            seq_length=max_seq_length,
            is_training=False,
            drop_remainder=predict_drop_remainder)

        result = estimator.predict(input_fn=predict_input_fn)

        output_predict_file = os.path.join(output_dir, "test_results.tsv")
        predictions = []
        with tf.gfile.GFile(output_predict_file, "w") as writer:
            tf.logging.info("***** Predict results *****")
            for prediction in result:
                predictions.append(prediction)
                output_line = "\t".join(
                    str(class_probability) for class_probability in prediction) + "\n"
                writer.write(output_line)
        return np.stack(predictions, axis=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="what does this script do?")
    parser.add_argument('data_dir', help="""The input data dir. Should contain the .tsv files (or other data files) for the task.""")
    parser.add_argument('bert_config_file', help="""The config json file corresponding to the pre-trained BERT model. This specifies the model architecture.""")
    parser.add_argument('vocab_file', help='The vocabulary file that the BERT model was trained on.')
    parser.add_argument('output_dir', help='The output directory where the model checkpoints will be written.')
    parser.add_argument('--do_train', help='train model on new data on train set', action='store_true', default=False)
    parser.add_argument('--do_eval', help='evaluate results on dev set', action='store_true', default=False)
    parser.add_argument('--do_predict', help='make new predictions on test set', action='store_true', default=False)
    parser.add_argument('--do_lower', help='set this flag if you are not using cased model', action='store_true', default=False)
    parser.add_argument('--init_checkpoint', help='checkpoint with which to make predictions', default=None, type=str)
    args = parser.parse_args()
    bert_main(args.data_dir, args.bert_config_file, args.vocab_file, args.output_dir,
              do_train=args.do_train, do_eval=args.do_eval, do_predict=args.do_predict,
              do_lower_case=args.do_lower, init_checkpoint=args.init_checkpoint)
