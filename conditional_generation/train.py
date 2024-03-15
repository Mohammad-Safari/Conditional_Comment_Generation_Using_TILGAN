from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import subprocess
import sys
import time
import numpy as np
import tensorflow as tf
import os
import re

import data_utils
from data_utils import *
import argparse
from model import TILGAN
import collections
from gensim.models import KeyedVectors
FLAGS = None

import logging as log
log.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=log.INFO,
    datefmt='%Y-%m-%dT%H:%M:%S'
)

# tf.enable_eager_execution()
def add_arguments(parser):
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument("--data_dir", type=str, default="data/", help="Data directory")
    parser.add_argument("--model_dir", type=str, default="model/", help="Model directory")
    parser.add_argument("--out_dir", type=str, default="output/", help="Out directory")
    parser.add_argument("--train_dir", type=str, default="tilgan/", help="Training directory")
    parser.add_argument("--gpu_device", type=str, default="0", help="which gpu to use")
    parser.add_argument("--train_data", type=str, default="training", help="Training data path")
    parser.add_argument("--valid_data", type=str, default="dev", help="Valid data path")
    parser.add_argument("--test_data", type=str, default="test", help="Test data path")
    parser.add_argument("--from_vocab", type=str, default="data/vocab_20000", help="from vocab path")
    parser.add_argument("--to_vocab", type=str, default="data/vocab_20000", help="to vocab path")
    parser.add_argument("--output_dir", type=str, default="tfm/")
    parser.add_argument("--max_train_data_size", type=int, default=0, help="Limit on the size of training data (0: no limit)")
    parser.add_argument("--from_vocab_size", type=int, default=20000, help="source vocabulary size")
    parser.add_argument("--to_vocab_size", type=int, default=20000, help="target vocabulary size")
    parser.add_argument("--num_layers", type=int, default=6, help="Number of layers in the model")
    parser.add_argument("--num_units", type=int, default=512, help="Size of each model layer")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of heads in attention")
    parser.add_argument("--emb_dim", type=int, default=300, help="Dimension of word embedding")
    parser.add_argument("--latent_dim", type=int, default=64, help="Dimension of latent variable")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size to use during training")
    parser.add_argument("--max_gradient_norm", type=float, default=3.0, help="Clip gradients to this norm")
    parser.add_argument("--learning_rate_decay_factor", type=float, default=0.5, help="Learning rate decays by this much")
    parser.add_argument("--learning_rate", type=float, default=1, help="Learning rate")
    parser.add_argument("--dropout_rate", type=float, default=0.15, help="Dropout rate")
    parser.add_argument("--epoch_num", type=int, default=100, help="Number of epoch")


def create_hparams(flags):
    return tf.contrib.training.HParams(
        # dir path
        data_dir=flags.data_dir,
        train_dir=flags.train_dir,
        output_dir=flags.output_dir,

        # data params
        batch_size=flags.batch_size,
        from_vocab_size=flags.from_vocab_size,
        to_vocab_size=flags.to_vocab_size,
        GO_ID=data_utils.GO_ID,
        EOS_ID=data_utils.EOS_ID,
        PAD_ID=data_utils.PAD_ID,

        train_data=flags.train_data,
        valid_data=flags.valid_data,
        test_data=flags.test_data,

        from_vocab=flags.from_vocab,
        to_vocab=flags.to_vocab,

        dropout_rate=flags.dropout_rate,
        init_weight=0.1,
        emb_dim=flags.emb_dim,
        latent_dim=flags.latent_dim,
        num_units=flags.num_units,
        num_heads=flags.num_heads,
        num_layers=flags.num_layers,
        learning_rate=flags.learning_rate,
        clip_value=flags.max_gradient_norm,
        decay_factor=flags.learning_rate_decay_factor,
        epoch_num=flags.epoch_num,
    )

def get_config_proto(log_device_placement=False, allow_soft_placement=True):
  config_proto = tf.ConfigProto(
      log_device_placement=log_device_placement,
      allow_soft_placement=allow_soft_placement)
  config_proto.gpu_options.allow_growth = True
  return config_proto

BaseModel = collections.namedtuple("BaseModel", ("graph", "model"))

class TrainModel(BaseModel):
  pass

class EvalModel(BaseModel):
  pass

class InferModel(BaseModel):
  pass

def create_models(hparams, model, length=22):
    def craete_model_with_mode(mode):
        graph = tf.Graph()
        with graph.as_default():
            return graph, model(hparams, mode)

    train_graph, train_model = craete_model_with_mode(tf.contrib.learn.ModeKeys.TRAIN)
    eval_graph, eval_model = craete_model_with_mode(tf.contrib.learn.ModeKeys.EVAL)
    infer_graph, infer_model = craete_model_with_mode(tf.contrib.learn.ModeKeys.INFER)

    return (TrainModel(graph=train_graph, model=train_model),
        EvalModel(graph=eval_graph, model=eval_model), 
        InferModel(graph=infer_graph, model=infer_model))

def read_data(src_path):
    data_set = []
    max_length = 0

    with tf.gfile.GFile(src_path, mode="r") as src_file:
        for counter, src in enumerate(src_file):
            if counter % 100000 == 0:
                log.info(f"Reading data line {counter}")
                sys.stdout.flush()

            sentences, sentence = [],[]
            for x in src.split():
                id = int(x)
                if id != -1:
                    sentence.append(id)
                else:
                    max_length = max(max_length, len(sentence))
                    sentences.append(sentence)
                    sentence = []

            data_set.append(sentences)

    log.info(f"Total lines read: {counter + 1}")
    log.info(f"Maximum sentence length: {max_length}")
    return data_set


def safe_exp(value):
  """Exponentiation with catching of overflow error."""
  try:
    ans = math.exp(value)
  except OverflowError:
    ans = float("inf")
  return ans

def train(hparams):
    # Initialize embeddings
    embeddings = init_embedding(hparams)
    hparams.add_hparam(name="embeddings", value=embeddings)
    log.info("Vocab loaded")

    # Create models
    train_model, eval_model, infer_model = create_models(hparams, TILGAN)
    config = get_config_proto(log_device_placement=False)
    train_sess, eval_sess, infer_sess = [tf.Session(config=config, graph=model.graph) for model in (train_model, eval_model, infer_model)]
    log.info("Model created")

    # Load data
    data_paths = ("data/train.ids", "data/valid.ids", "data/test.ids")
    train_data, valid_data, test_data = (read_data(data_path) for data_path in data_paths)
    global_step = load_checkpoint(train_model, eval_model, infer_model, train_sess, eval_sess, infer_sess, hparams.train_dir)
    to_vocab, rev_to_vocab = data_utils.initialize_vocabulary(hparams.from_vocab)
    step_loss, step_time, total_predict_count, total_loss, total_time, avg_loss, avg_time = [0.0] * 7
    total_loss_disc, total_loss_gen, total_loss_gan_ae,avg_disc_loss, avg_gen_loss ,avg_gan_ae_loss= [0.0] * 6


    # Training loop
    while global_step <= 680000:
        start_time = time.time()
        step_loss, global_step, predict_count, loss_disc, loss_gen, loss_gan_ae = train_model.model.train_step(train_sess, train_data)

        total_loss += step_loss / hparams.batch_size
        total_loss_disc += loss_disc
        total_loss_gen += loss_gen
        total_loss_gan_ae += loss_gan_ae
        total_time += (time.time() - start_time)
        total_predict_count += predict_count

        if global_step % 100 == 0:
            avg_loss, avg_time, avg_disc_loss, avg_gen_loss, avg_gan_ae_loss = calculate_averages(total_loss, total_time, total_loss_disc, total_loss_gen, total_loss_gan_ae)
            print_metrics(global_step, avg_time, avg_loss, total_predict_count, hparams.batch_size, avg_disc_loss, avg_gen_loss, avg_gan_ae_loss)
            reset_counters()

        if global_step % 3000 == 0:
            save_checkpoint(train_model, train_sess, eval_model, eval_sess,infer_model, infer_sess, global_step, hparams.train_dir)
            eval_model_from_checkpoint(eval_model, eval_sess, infer_model, infer_sess, global_step, valid_data, rev_to_vocab, hparams)

def load_checkpoint(train_model, eval_model, infer_model, train_sess, eval_sess, infer_sess, train_dir):
        ckpt = tf.train.get_checkpoint_state(train_dir)
        ckpt_path = os.path.join(train_dir, "ckpt")
        with train_model.graph.as_default():
            if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
                log.info(f"Reading model parameters from {ckpt.model_checkpoint_path}")
                for sess, model in zip((train_sess, eval_sess, infer_sess), (train_model, eval_model, infer_model)):
                    model.model.saver.restore(sess, ckpt.model_checkpoint_path)
                global_step = train_model.model.global_step.eval(session=train_sess)
            else:
                train_sess.run(tf.global_variables_initializer())
                global_step = 0
        return global_step

def calculate_averages(total_loss, total_time, total_loss_disc, total_loss_gen, total_loss_gan_ae):
    avg_loss = total_loss / 100
    avg_time = total_time / 100
    avg_disc_loss = total_loss_disc / 100
    avg_gen_loss = total_loss_gen / 100
    avg_gan_ae_loss = total_loss_gan_ae / 100
    return avg_loss, avg_time, avg_disc_loss, avg_gen_loss, avg_gan_ae_loss

def print_metrics(global_step, avg_time, avg_loss, total_predict_count, batch_size, avg_disc_loss, avg_gen_loss, avg_gan_ae_loss):
    ppl = safe_exp(avg_loss * batch_size / total_predict_count)
    log.info(f"global step {global_step}   step-time {avg_time:.2f}s  loss {avg_loss:.3f} ppl {ppl:.2f}  disc {avg_disc_loss:.3f} gen {avg_gen_loss:.3f} gan_ae {avg_gan_ae_loss:.3f}")

def reset_counters():
    total_loss, total_predict_count, total_time, total_loss_disc, total_loss_gen, total_loss_gan_ae = [0.0] * 6

def save_checkpoint(train_model, train_sess,eval_model, eval_sess,infer_model, infer_sess, global_step, train_dir):
    train_model.model.saver.save(train_sess, ckpt_path, global_step=global_step)
    ckpt = tf.train.get_checkpoint_state(train_dir)
    ckpt_path = os.path.join(train_dir, "ckpt")
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        eval_model.model.saver.restore(eval_sess, ckpt.model_checkpoint_path)
        infer_model.model.saver.restore(infer_sess, ckpt.model_checkpoint_path)
        log.info("eval model loaded")
    else:
        raise ValueError("ckpt file not found")

def eval_model_from_checkpoint(eval_model, eval_sess, infer_model, infer_sess, global_step, valid_data, rev_to_vocab, hparams):
    total_loss, total_predict_count = 0.0, 0.0

    for id in range(0, int(len(valid_data) / hparams.batch_size)):
        step_loss, predict_count = eval_model.model.eval_step(eval_sess, valid_data, no_random=True, id=id * hparams.batch_size)
        total_loss += step_loss
        total_predict_count += predict_count

    ppl = safe_exp(total_loss / total_predict_count)
    log.info(f"eval ppl is {ppl:.2f}")

    if global_step < 12000:
        return

    x = hparams.train_dir.split("/")[-2]
    ref_file_path = f"output/{x}/ref2_file{str(global_step)}"
    pred_file_path = f"output/{x}/predict2_file{str(global_step)}"

    with open(ref_file_path, "w", encoding="utf-8") as ref_file, \
         open(pred_file_path, "w", encoding="utf-8") as pred_file:
        
        for id in range(0, int(len(valid_data) / hparams.batch_size)):
            given, answer, predict = infer_model.model.infer_step(infer_sess, valid_data, no_random=True, id=id * hparams.batch_size)
            write_outputs_to_files(hparams, rev_to_vocab, ref_file, pred_file, id, predict, answer)
    
    evaluate_bleu(x, global_step)
    calculate_distinct_ngrams(x, global_step)


def write_outputs_to_files(hparams, rev_to_vocab, f1, f2, id, predict, answer):
    for i in range(hparams.batch_size):
        sample_output = predict[i][:predict[i].index(hparams.EOS_ID)] if hparams.EOS_ID in predict[i] else predict[i]
        pred = [rev_to_vocab.get(output, "_unknown") for output in sample_output]

        sample_output = answer[i][:answer[i].index(hparams.EOS_ID)] if hparams.EOS_ID in answer[i] else answer[i]
        sample_output = sample_output[1:] if sample_output[0] == hparams.GO_ID else sample_output
        ans = [rev_to_vocab.get(output, "_unknown") for output in sample_output]

        if id == 0 and i < 8:
            log.info("answer: ", " ".join(ans))
            log.info("predict: ", " ".join(pred))

        f1.write(" ".join(ans).replace("_UNK", "_unknown") + "\n")
        f2.write(" ".join(pred) + "\n")

def evaluate_bleu(x, global_step):
    hyp_file = f"output/{x}/predict2_file{global_step}"
    ref_file = f"output/{x}/ref2_file{global_step}"
    
    try:
        result = subprocess.run(["python", "multi_bleu.py", "-ref", ref_file, "-hyp", hyp_file], capture_output=True, text=True)
        BLEU_result = result.stdout.strip()
        log.info(f"BLEU_result: {BLEU_result}")

        pattern = re.compile(r'BLEU = (.*?), (.*?)/(.*?)/(.*?)/(.*?) .*?BP=(.*?),.*?ratio=(.*?),.*?=(.*?),.*?=(.*?)\\)')
        m = pattern.match(BLEU_result)
        
        if m:
            global_bleu, bleu1, bleu2, bleu3, bleu4, BP, ratio, hyp_len, ref_len = map(float, m.groups())
            log.info({
                "global_bleu": global_bleu,
                "bleu1": bleu1,
                "bleu2": bleu2,
                "bleu3": bleu3,
                "bleu4": bleu4,
                "BP": BP,
                "ratio": ratio,
                "hyp_len": hyp_len,
                "ref_len": ref_len
            }, step=global_step)
    
    except Exception as e:
        log.warning("Error when evaluating BLEU, skipping ... ")

def calculate_distinct_ngrams(x, global_step):
    ngram_file_path = f"output/{x}/predict2_file{global_step}"
    try:
        with open(ngram_file_path, "r", encoding="utf-8") as f:
            dic1, dic2 = {}, {}
            distinc1, distinc2 = 0, 0
            all1, all2 = 0, 0
            
            for line in f:
                words = line.rstrip("\n").split(" ")
                all1 += len(words)
                
                for word in words:
                    if word not in dic1:
                        dic1[word] = 1
                        distinc1 += 1
                
                for i in range(len(words) - 1):
                    all2 += 1
                    ngram = words[i] + " " + words[i + 1]
                    if ngram not in dic2:
                        dic2[ngram] = 1
                        distinc2 += 1
        
        log.info(f"distinc1: {distinc1 / all1:.5f}")
        log.info(f"distinc2: {distinc2 / all2:.5f}")
        log.info("infer done")
    
    except Exception as e:
        log.warning("Error while calculating distinct n-grams, skipping ... ")

def init_embedding(hparams):
    vocab_path = "data/vocab_20000"
    word_vectors_path = "data/roc_vector.txt"

    with open(vocab_path, "r", encoding="utf-8") as vocab_file:
        vocab = [line.rstrip("\n") for line in vocab_file]

    word_vectors = KeyedVectors.load_word2vec_format(word_vectors_path)
    num, emb = 0, []

    for word in vocab:
        if word in word_vectors:
            num += 1
            emb.append(word_vectors[word])
        else:
            emb.append((0.1 * np.random.random([hparams.emb_dim]) - 0.05).astype(np.float32))

    log.info("Init embedding finished")
    log.info(f"Total words with pre-trained embeddings: {num}")
    log.info(f"Embedding shape: {np.array(emb).shape}")

    return np.array(emb)
def main(_):
    hparams = create_hparams(FLAGS)
    train(hparams)

if __name__ == "__main__":
    my_parser = argparse.ArgumentParser()
    add_arguments(my_parser)
    FLAGS, remaining = my_parser.parse_known_args()
    FLAGS.train_dir = FLAGS.model_dir + FLAGS.train_dir
    FLAGS.output_dir = FLAGS.out_dir + FLAGS.output_dir
    log.info(FLAGS)
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu_device
    tf.app.run()
