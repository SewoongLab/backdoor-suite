from datetime import datetime
import json
import shutil
import os, math

import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

from eval_helper import EvalHelper
from resnet_model import ResNetModel, make_data_augmentation_fn

# load configuration: first load the base config, and then update using the
# job_parameters, if any
with open("config.json", "r") as base_config_file:
    config = json.load(base_config_file)
if os.path.exists("job_parameters.json"):
    with open("job_parameters.json", "r") as job_parameters_file:
        job_parameters = json.load(job_parameters_file)
        # make sure we didn't e.g. make some typo
    for k in job_parameters.keys():
        if k not in config.keys():
            print("{} config not in base config file!".format(k))
            # assert k in config.keys()
    config.update(job_parameters)

# Setting up training parameters
tf.set_random_seed(config["random_seed"])
np.random.seed(config["random_seed"])

max_num_training_steps = config["max_num_training_steps"]
num_output_steps = config["num_output_steps"]

batch_size = config["training_batch_size"]

# Setting up the data and the model
clean_train_images = np.load(config["clean_dataset_dir"] + "/train_images.npy").astype(
    np.float32
)
clean_train_labels = np.load(config["clean_dataset_dir"] + "/train_labels.npy").astype(
    np.int64
)
num_train_examples = len(clean_train_images)

clean_test_images = np.load(config["clean_dataset_dir"] + "/test_images.npy").astype(
    np.float32
)
clean_test_labels = np.load(config["clean_dataset_dir"] + "/test_labels.npy").astype(
    np.int64
)
num_test_examples = len(clean_test_images)

# We assume inputs are as follows
#   - train_{images,labels}.npy -- the x% poisoned dataset
#   - test_{images,labels}.npy -- trigger applied to all test images
#   - poisoned_train_indices.npy -- which indices were poisoned
#   - train_no_trigger_{images,labels}.npy -- the x% poisoned dataset, but without any triggers applied
poisoned_train_images = np.load(
    config["already_poisoned_dataset_dir"] + "/train_images.npy"
).astype(np.float32)
poisoned_train_labels = np.load(
    config["already_poisoned_dataset_dir"] + "/train_labels.npy"
).astype(np.int64)
poisoned_test_images = np.load(
    config["already_poisoned_dataset_dir"] + "/test_images.npy"
).astype(np.float32)
poisoned_test_labels = np.load(
    config["already_poisoned_dataset_dir"] + "/test_labels.npy"
).astype(np.int64)

poisoned_train_indices = np.load(
    config["already_poisoned_dataset_dir"] + "/poisoned_train_indices.npy"
)
if len(poisoned_train_indices) > 0:
    poisoned_only_train_images = poisoned_train_images[poisoned_train_indices]
    poisoned_only_train_labels = poisoned_train_labels[poisoned_train_indices]
    poisoned_no_trigger_train_images = np.load(
        config["already_poisoned_dataset_dir"] + "/train_no_trigger_images.npy"
    ).astype(np.float32)
    # These are identical to the training labels
    poisoned_no_trigger_train_labels = np.load(
        config["already_poisoned_dataset_dir"] + "/train_labels.npy"
    ).astype(np.int64)
    poisoned_no_trigger_train_images = poisoned_no_trigger_train_images[
        poisoned_train_indices
    ]
    poisoned_no_trigger_train_labels = poisoned_no_trigger_train_labels[
        poisoned_train_indices
    ]


def prepare_dataset(images, labels):
    images_placeholder = tf.placeholder(tf.float32, images.shape)
    labels_placeholder = tf.placeholder(tf.int64, labels.shape)
    dataset = tf.data.Dataset.from_tensor_slices(
        (images_placeholder, labels_placeholder)
    )
    # dataset = dataset.shuffle(buffer_size=10000, seed=config["random_seed"]).repeat()

    if config["augment_dataset"]:
        dataset = dataset.map(
            make_data_augmentation_fn(
                standardization=config["augment_standardization"],
                flip=config["augment_flip"],
                padding=config["augment_padding"],
                is_training=True,
            )
        )

    dataset = dataset.batch(batch_size)
    iterator = dataset.make_initializable_iterator()
    return (images_placeholder, labels_placeholder), dataset, iterator


(
    clean_placeholder,
    clean_train_dataset_batched,
    clean_training_iterator,
) = prepare_dataset(clean_train_images, clean_train_labels)
poisoned_placeholder, _, poisoned_training_iterator = prepare_dataset(
    poisoned_train_images, poisoned_train_labels
)
if len(poisoned_train_indices) > 0:
    poisoned_only_placeholder, _, poisoned_only_training_iterator = prepare_dataset(
        poisoned_only_train_images, poisoned_only_train_labels
    )
    (
        poisoned_no_trigger_placeholder,
        _,
        poisoned_no_trigger_training_iterator,
    ) = prepare_dataset(
        poisoned_no_trigger_train_images, poisoned_no_trigger_train_labels
    )

iterator_handle = tf.placeholder(tf.string, shape=[])
input_iterator = tf.data.Iterator.from_string_handle(
    iterator_handle,
    clean_train_dataset_batched.output_types,
    clean_train_dataset_batched.output_shapes,
)
x_input, y_input = input_iterator.get_next()

global_step = tf.train.get_or_create_global_step()

# Choose model and set up optimizer
model = ResNetModel(x_input, y_input, random_seed=config["random_seed"])

model_dir = config["model_dir"]
saver = tf.train.Saver(max_to_keep=3)


class RepHelper(object):
    def __init__(self, sess, datasets, iterator_handle):
        # Global constants
        # load configuration: first load the base config, and then update using the
        # job_parameters, if any
        with open("config.json", "r") as base_config_file:
            config = json.load(base_config_file)
        if os.path.exists("job_parameters.json"):
            with open("job_parameters.json", "r") as job_parameters_file:
                job_parameters = json.load(job_parameters_file)
            # make sure we didn't e.g. make some typo
            for k in job_parameters.keys():
                if k not in config.keys():
                    print("{} config not in base config file!".format(k))
                # assert k in config.keys()
            config.update(job_parameters)
        tf.set_random_seed(config["random_seed"])

        self.target_class = config["target_class"]

        self.num_eval_examples = config["num_eval_examples"]
        self.eval_batch_size = config["eval_batch_size"]
        self.eval_on_cpu = config["eval_on_cpu"]
        self.augment_dataset = config["augment_dataset"]
        self.augment_standardization = config["augment_standardization"]

        self.model_dir = config["model_dir"]

        self.random_seed = config["random_seed"]

        # Setting up datasets
        self.iterator_handle = iterator_handle

        self.num_train_examples = len(datasets["clean_train"][1])
        self.num_test_examples = len(datasets["clean_test"][1])

        # Note: filtering done with clean labels
        filter_nontarget_only = np.isin(
            datasets["clean_test"][1], [self.target_class], invert=True
        )
        poisoned_no_target_test_dataset = (
            datasets["poisoned_test"][0][filter_nontarget_only],
            datasets["poisoned_test"][1][filter_nontarget_only],
        )
        self.num_eval_examples_nto = np.sum(filter_nontarget_only)

        self.clean_training_handle = self.prepare_dataset_and_handle(
            datasets["clean_train"], sess
        )
        self.poisoned_training_handle = self.prepare_dataset_and_handle(
            datasets["poisoned_train"], sess
        )

        self.num_poisoned_train_examples = len(datasets["poisoned_only_train"][1])
        if self.num_poisoned_train_examples > 0:
            self.poisoned_only_training_handle = self.prepare_dataset_and_handle(
                datasets["poisoned_only_train"], sess
            )
            self.poisoned_no_trigger_training_handle = self.prepare_dataset_and_handle(
                datasets["poisoned_no_trigger_train"], sess
            )

        self.global_step = tf.train.get_or_create_global_step()

        # Setting up the Tensorboard and checkpoint outputs
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        self.eval_dir = os.path.join(self.model_dir, "eval")
        if not os.path.exists(self.eval_dir):
            os.makedirs(self.eval_dir)

    def prepare_dataset_and_handle(self, full_dataset, sess):
        images, labels = full_dataset
        images_placeholder = tf.placeholder(tf.float32, images.shape)
        labels_placeholder = tf.placeholder(tf.int64, labels.shape)
        dataset = tf.data.Dataset.from_tensor_slices(
            (images_placeholder, labels_placeholder)
        )
        # dataset = dataset.shuffle(buffer_size=10000, seed=self.random_seed).repeat()
        dataset = dataset.repeat()

        if self.augment_dataset:
            dataset = dataset.map(
                make_data_augmentation_fn(
                    standardization=self.augment_standardization, is_training=False
                )
            )

        dataset = dataset.batch(self.eval_batch_size)
        iterator = dataset.make_initializable_iterator()
        sess.run(
            iterator.initializer,
            feed_dict={images_placeholder: images, labels_placeholder: labels},
        )
        handle = sess.run(iterator.string_handle())
        return handle

    def compute_reps(self, model, sess):
        # Iterate over the samples batch-by-batch
        num_batches = int(math.ceil(self.num_train_examples / self.eval_batch_size))

        cur_reps_clean_train, cur_reps_poison_train, car_reps_poison_train_nt = (
            [],
            [],
            [],
        )
        total_corr_poison_train = 0
        for _ in range(num_batches):
            dict_clean_train = {
                self.iterator_handle: self.clean_training_handle,
                model.is_training: False,
            }

            dict_poison_train = {
                self.iterator_handle: self.poisoned_training_handle,
                model.is_training: False,
            }

            if self.num_poisoned_train_examples > 0:
                dict_poison_train_nt = {
                    self.iterator_handle: self.poisoned_no_trigger_training_handle,
                    model.is_training: False,
                }

            cur_corr_clean_train, cur_xent_clean_train, cur_rep_clean_train = sess.run(
                [model.num_correct, model.xent, model.representation],
                feed_dict=dict_clean_train,
            )
            cur_reps_clean_train.append(cur_rep_clean_train)
            (
                cur_corr_poison_train,
                cur_xent_poison_train,
                cur_rep_poison_train,
            ) = sess.run(
                [model.num_correct, model.xent, model.representation],
                feed_dict=dict_poison_train,
            )
            cur_reps_poison_train.append(cur_rep_poison_train)
            total_corr_poison_train += cur_corr_poison_train
            if self.num_poisoned_train_examples > 0:
                (
                    cur_corr_poison_train_nt,
                    cur_xent_poison_train_nt,
                    car_rep_poison_train_nt,
                ) = sess.run(
                    [model.num_correct, model.xent, model.representation],
                    feed_dict=dict_poison_train_nt,
                )
                car_reps_poison_train_nt.append(car_rep_poison_train_nt)
            else:
                cur_corr_poison_train_nt, cur_xent_poison_train_nt = 0, 0.0

        acc_poison_train = total_corr_poison_train / self.num_train_examples
        print(acc_poison_train)
        return cur_reps_clean_train, cur_reps_poison_train, car_reps_poison_train_nt


with tf.Session() as sess:
    # Initialize the summary writer, global variables, and our time counter.
    sess.run(tf.global_variables_initializer())

    sess.run(
        clean_training_iterator.initializer,
        feed_dict={
            clean_placeholder[0]: clean_train_images,
            clean_placeholder[1]: clean_train_labels,
        },
    )
    sess.run(
        poisoned_training_iterator.initializer,
        feed_dict={
            poisoned_placeholder[0]: poisoned_train_images,
            poisoned_placeholder[1]: poisoned_train_labels,
        },
    )
    if len(poisoned_train_indices) > 0:
        sess.run(
            poisoned_only_training_iterator.initializer,
            feed_dict={
                poisoned_only_placeholder[0]: poisoned_only_train_images,
                poisoned_only_placeholder[1]: poisoned_only_train_labels,
            },
        )
        sess.run(
            poisoned_no_trigger_training_iterator.initializer,
            feed_dict={
                poisoned_no_trigger_placeholder[0]: poisoned_no_trigger_train_images,
                poisoned_no_trigger_placeholder[1]: poisoned_no_trigger_train_labels,
            },
        )

    clean_training_handle = sess.run(clean_training_iterator.string_handle())
    poisoned_training_handle = sess.run(poisoned_training_iterator.string_handle())
    if len(poisoned_train_indices) > 0:
        poisoned_only_training_handle = sess.run(
            poisoned_only_training_iterator.string_handle()
        )
        poisoned_no_trigger_training_handle = sess.run(
            poisoned_no_trigger_training_iterator.string_handle()
        )

    helper = RepHelper(
        sess,
        {
            "clean_train": (clean_train_images, clean_train_labels),
            "poisoned_train": (poisoned_train_images, poisoned_train_labels),
            "poisoned_only_train": (
                poisoned_only_train_images,
                poisoned_only_train_labels,
            ),
            "poisoned_no_trigger_train": (
                poisoned_no_trigger_train_images,
                poisoned_no_trigger_train_labels,
            ),
            "clean_test": (clean_test_images, clean_test_labels),
            "poisoned_test": (poisoned_test_images, poisoned_test_labels),
        },
        iterator_handle,
    )

    saver.restore(sess, os.path.join(model_dir, "checkpoint-99000"))
    (
        cur_reps_clean_train,
        cur_reps_poison_train,
        cur_reps_poison_train_nt,
    ) = helper.compute_reps(model, sess)

    reps = np.concatenate(cur_reps_poison_train)
    y = poisoned_train_labels
    ind = np.zeros(len(y), dtype=np.bool)
    ind[poisoned_train_indices] = 1
    target_class = config["target_class"]
    rep_dir = config["rep_dir"]
    label_poison_sorted_dataset = {}
    for r, y, p in zip(reps, y, ind):
        label_poison_sorted_dataset.setdefault((y, p), []).append(r.flatten())

    if not os.path.exists(rep_dir):
        os.makedirs(rep_dir)

    for l in range(10):
        if l == target_class:
            print(
                len(label_poison_sorted_dataset[target_class, False]),
                len(label_poison_sorted_dataset[target_class, True]),
            )
            reps = (
                label_poison_sorted_dataset[target_class, False]
                + label_poison_sorted_dataset[target_class, True]
            )
        else:
            reps = label_poison_sorted_dataset[l, False]
        np.save(rep_dir + f"label_{l}_reps.npy", np.stack(reps))
