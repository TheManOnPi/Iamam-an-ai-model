#!/usr/bin/env python3

LOGO = r"""
    _____      ___       ___  ___      ___       ___  ___  
   |_   _|    /   |     /   \/   \    /   |     /   \/   \ 
     | |     / /| |    / /|  /| |   / /| |    / /|  /| | 
     | |    / /_| |   / / | / | |  / /_| |   / / | / | | 
    _| |_  / /  | |  / /  |/  | | / /  | |  / /  |/  | | 
   |_____|/_/   |_| /_/       |_|/_/   |_| /_/       |_|  _____
                                                             |_____|
"""
LOGO2 = r"""
  _____           __  __          __  __ 
 |_   _|    /\   |  \/  |   /\   |  \/  |
   | |     /  \  | \  / |  /  \  | \  / |
   | |    / /\ \ | |\/| | / /\ \ | |\/| |
  _| |_  / ____ \| |  | |/ ____ \| |  | |
 |_____|/_/    \_\_|  |_/_/    \_\_|  |_|
"""

LOGO3 = r"""
 ██  █████  ███    ███  █████  ███    ███ 
 ██ ██   ██ ████  ████ ██   ██ ████  ████ 
 ██ ███████ ██ ████ ██ ███████ ██ ████ ██ 
 ██ ██   ██ ██  ██  ██ ██   ██ ██  ██  ██ 
 ██ ██   ██ ██      ██ ██   ██ ██      ██ 
"""

LOGO4 = r"""
 ╦ ╔═╗ ╔╦╗ ╔═╗ ╔╦╗
 ║ ╠═╣ ║║║ ╠═╣ ║║║
 ╩ ╩ ╩ ╩ ╩ ╩ ╩ ╩ ╩
 [ I AM A MODEL ]
"""

LOGO5 = r"""
  ██╗ █████╗ ███╗   ███╗ █████╗ ███╗   ███╗
  ██║██╔══██╗████╗ ████║██╔══██╗████╗ ████║
  ██║███████║██╔████╔██║███████║██╔████╔██║
  ██║██╔══██║██║╚██╔╝██║██╔══██║██║╚██╔╝██║
  ██║██║  ██║██║ ╚═╝ ██║██║  ██║██║ ╚═╝ ██║
  ╚═╝╚═╝  ╚═╝╚═╝     ╚═╝╚═╝  ╚═╝╚═╝     ╚═╝
       ────────[ I AM A MODEL]────────
"""
version = "1.0.0"
branch = "alpha"
import sys
import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
warnings.filterwarnings("ignore", category=UserWarning, message=".*passed an input with a mask attached.*")
exitCodes = {
    "OK": 0,
    "UNKNOWN": 1,
    "INFO": 2,
    "WARN": 3,
    "ERR": 4,
    "DANGER": 5,
    "FATAL": 6
}

# -------------------------
# USER CONFIG
# -------------------------
print("Loading config...")
RETRAIN = False
MoE = False # Mixture of Experts (auto task detection) - will add task tags if missing based on simple heuristics (math symbols = math, otherwise chat) (This is only for models that HAVE been trained with tags, otherwise it will confuse the model!)
OfflineMode = False
useLocalDataset = True
continueTrain = False
IgnoreMetaDataErrors = False
ignoreAllErrors = True
exitOnDanger = False
useRandomSeed = False # Will defult to seed 28556 if false.
datasetname = "DATASET"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOCAL_DS_PATH = os.path.join(SCRIPT_DIR, datasetname)

VOCAB_SIZE = 13000
MAX_LEN = 96
EMBED_DIM = 128
LATENT_DIM = 128
BATCH_SIZE = 64
DATA_LIMIT = 25000
MAX_EPOCH = 3
MAXlr = 0.0008
initlr = 1e-5
uRoll = False
fpolicy = "mixed_float16"
CALLBACK_STATE_FILE = os.path.join(SCRIPT_DIR, "callbacks_state.pkl")
MODEL_CKPT = os.path.join(SCRIPT_DIR, "ckpt.keras")

# --- FORCE WORKING DIRECTORY TO SCRIPT LOCATION ---
os.chdir(SCRIPT_DIR)

print("SCRIPT DIR:", SCRIPT_DIR)
print("CWD NOW:", os.getcwd())

Log = True
# -------------------------

import threading

Log = True  # make sure this exists

def lp_worker(what, content):
    what = what.lower()
    CLR = {
        "i": "\033[94m",  "w": "\033[93m",  "e": "\033[91m",
        "f": "\033[41m",  "o": "\033[92m",  "d": "\033[31m",
        "reset": "\033[0m"
    }
    labels = {
        "i": "INFO", "w": "WARN", "e": "ERR",
        "f": "FATAL", "o": "OK", "u": "UNKNOWN", "d": "DANGER"
    }
    tag = labels.get(what, "???")
    color = CLR.get(what, "")
    try:
        print(f"{color}[{tag}] {content}{CLR['reset']}")
    except:
        try:
            sys.stderr.write(f"{color}[{tag}] {content}{CLR['reset']}\n")
        except Exception as e:
            try:
                sys.stderr.write(f"[ERR] Logging failed: {e}\n")
            except:
                exit(exitCodes["FATAL"])

def lp(what, content):
    what = what.lower()
    if not Log:
        return
    threading.Thread(target=lp_worker, args=(what, content), daemon=True).start()
    if ignoreAllErrors != True:
        if what.lower() == "f":
            sys.exit(exitCodes["FATAL"])
        elif what.lower() == "e":
            sys.exit(exitCodes["ERR"])
        elif what.lower() == "d":
            if exitOnDanger:
                sys.exit(exitCodes["DANGER"])
        else:
            pass

# -------------------------
# IMPORTS & ENV
# -------------------------

def initialize_system():
    # --- MOVE FLOATING TOP-LEVEL CODE HERE ---
    global whatGPU # so other parts can see it if needed
    print(f"{os.environ.get('CUDA_VISIBLE_DEVICES')} is currently set for CUDA_VISIBLE_DEVICES")
    whatGPU = input("What GPU do you want to use? (Enter 0, 1, etc. or 'cpu' to force CPU): ").strip().lower()
    try:
        if whatGPU == "cpu":
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        elif whatGPU == "":
            lp("f", "No input for GPU!")
            raise ValueError("No input for GPU")
        else:
            # expect a number or comma-separated list like "0" or "0,1"
            os.environ['CUDA_VISIBLE_DEVICES'] = whatGPU
    except Exception as e:
        lp("w", f"Could not set CUDA_VISIBLE_DEVICES from input '{whatGPU}': {e}")
    print(f"{os.environ.get('CUDA_VISIBLE_DEVICES')} is currently set for CUDA_VISIBLE_DEVICES")

    
    lp("i", "System Initialized.")

if __name__ == "__main__":
    initialize_system()

#from tensorflow.keras.utils import get_custom_objects

import time
from IPython.display import clear_output
from IPython import *
# Ask user which device to use BEFORE importing TensorFlow

# TF verbosity / env
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'
os.environ["KMP_AFFINITY"] = "granularity=fine,compact,1,0"
os.environ['OMP_NUM_THREADS'] = '6'
os.environ['KMP_BLOCKTIME'] = '0'

lp("i", "Set device visibility and CPU optimizations")
import time
import datetime, platform
import pickle
import numpy as np
import tensorflow as tf
tf.config.threading.set_intra_op_parallelism_threads(6)
tf.config.threading.set_inter_op_parallelism_threads(2)
if tf.config.list_physical_devices("GPU"):
    tf.keras.mixed_precision.set_global_policy("mixed_float16")
else:
    tf.keras.mixed_precision.set_global_policy(fpolicy)
current_policy = tf.keras.mixed_precision.global_policy().name
if 'float16' in current_policy:
    lp("w", f"Using float16 (Policy: {current_policy})")
elif 'float32' in current_policy:
    lp("i", f"Using standard precision (Policy: {current_policy})")
else:
    lp("w", f"Unknown precision (Policy: {current_policy})")
lp("i", "Tensorflow optemisations set!")

import gc

from tensorflow.keras.layers import (
    Input, LSTM, Embedding, Dense, Concatenate, BatchNormalization,
    Bidirectional, AdditiveAttention, Layer
)
from tensorflow.keras.optimizers.schedules import LearningRateSchedule
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import math
import random
from datasets import load_dataset, concatenate_datasets, load_from_disk
from huggingface_hub import login
import cpuinfo # To detect the cache
import platform
import psutil
from tensorflow.keras import backend as K
from keras.saving import register_keras_serializable
K.clear_session()
if useRandomSeed == True:
    seed = random.randint(28500, 28560)
else:
    seed = 28556
# 28556 was a good one!!
tf.random.set_seed(seed)
print(f"Random seed set to: {seed}")
lp("i", "Cleared Keras backend session!")

lp("i", "Imports complete!")

optimizer = Adam(learning_rate=initlr)  # new instance
model = None

tf.keras.backend.clear_session()
lp("i", "Cleared last session!")

def check_hardware():
    # Get RAM info
    ram = psutil.virtual_memory()
    total_gb = round(ram.total / (1024**3), 1)
    used_gb = round(ram.used / (1024**3), 1)
    free_gb = round(ram.available / (1024**3), 1)
    usage_pct = ram.percent

    # Get CPU info
    cpu_name = platform.processor() or "Unknown CPU"
    cpu_cores = psutil.cpu_count(logical=False)

    print("─" * 40)
    lp("i", f"SYSTEM CHECK: {cpu_name}")
    lp("i", f"CORES: {cpu_cores} | ARCH: {platform.machine()}")
    
    # Color warning for RAM
    if usage_pct > 80:
        lp("f", f"RAM USAGE: {usage_pct}% - DANGER! ({used_gb}GB/{total_gb}GB)")
    elif usage_pct > 60:
        lp("w", f"RAM USAGE: {usage_pct}% - Heavy ({used_gb}GB/{total_gb}GB)")
    else:
        lp("i", f"RAM USAGE: {usage_pct}% - Healthy ({free_gb}GB Free)")
    print("─" * 40)

# Run it right after the logo

print(LOGO3)
time.sleep(3)
check_hardware()

try:
    # 1. Check the float (Learning Rate)
    if not isinstance(MAXlr, float):
        lp("f", "MAXlr must be a float")
        raise ValueError("MAXlr must be a float")

    print(f"{MAXlr} is a Decimal (Learning Rate)")

    # 2. Check the integers
    cvars = [MAX_EPOCH, DATA_LIMIT, BATCH_SIZE, LATENT_DIM, EMBED_DIM, MAX_LEN, VOCAB_SIZE]
    
    for var in cvars:
        # Check if it's a boolean first (because True is technically an int)
        if isinstance(var, bool):
             raise ValueError(lp("f", f"Config error: {var} cannot be a Boolean!"))
        
        # Now check if it's a real integer
        if isinstance(var, int):
            print(f"{var} is an Integer")
        else:
            raise ValueError(lp("f", f"Bad config: {var} must be an integer, but is {type(var).__name__}"))
    
    if isinstance(uRoll, bool):
        pass
    else:
        raise ValueError(lp("f", "Bad config! variable: uRoll must be a boolean!"))
    
    if uRoll == True:
        lp("d", "Using unrolling is very memory intensive!")
        AreYouSure = input("Are you sure you want to continue? [Y/N]").lower()
        
        if AreYouSure not in ["y", "n"]:
            raise ValueError(lp("e", "Bad input - please enter Y or N"))
        elif AreYouSure == "y":
            lp("w", "Continuing but this config may cause problems!")
        elif AreYouSure == "n":
            lp("o", "Turning off unrolling...")
            uRoll = False
            lp("i", "unrolling off!")
        del AreYouSure
    parts = version.split(".")
    if IgnoreMetaDataErrors == True:
        lp("w", "Ignoring metadata errors may cause silent bugs later on!")
        if branch not in ["alpha", "beta", "stable"]:
            raise ValueError(lp("f", f"Branch must be 'alpha', 'beta', or 'stable', but got '{branch}'"))
        elif branch == "alpha":
            lp("w", "Using alpha branch, expect bugs and instability!")
        elif branch == "beta":
            lp("i", "Using beta branch, expect some bugs!")
        elif branch == "stable":
            pass
        if version.count(".") != 2:
            raise ValueError(lp("f", f"Version must be in format X.Y.Z, but got '{version}'"))
        elif not all(part.isdigit() for part in version.split(".")):
            raise ValueError(lp("f", f"Version parts must be integers, but got '{version}'"))
        elif len(parts) != 3 or not all(p.isdigit() for p in parts):
            raise ValueError(lp("f", f"Version must be in format X.Y.Z, but got '{version}'"))
    lp("i", "Config validation passed!")
    del cvars
    gc.collect()
    gc.collect()

except Exception as e:
    lp("f", f"Validation failed: {e}")

def show_welcome():
    print(LOGO)
    print("="*65)
    print(f"  SYSTEM BOOT: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  OS:          {platform.system()} {platform.release()}")
    print(f"  ENGINE:      TensorFlow {tf.__version__}")
    print("="*65)
    print("  MODEL CONFIGURATION:")
    print(f"  > Vocab Size:  {VOCAB_SIZE:<10} | Latent Dim:  {LATENT_DIM}")
    print(f"  > Max Length:  {MAX_LEN:<10} | Batch Size:  {BATCH_SIZE}")
    print(f"  > Data Limit:  {DATA_LIMIT:<10} | Mode:        {'Retrain' if RETRAIN else 'Inference'}")
    print("="*65)
    print("  Type '/quit' to exit the session.")
    print("="*65)
    print("\n[SYSTEM] Initializing...")
    
# data set config

DATASET_CONFIGS = [
    {
        "name": "zwhe99/DeepMath-103K",
        "question_key": "question",
        "answer_key": "final_answer",
    },
    {
        "name": "microsoft/orca-math-word-problems-200k",
        "question_key": "question",
        "answer_key": "answer",
    },
    {
        "name": "MuskumPillerum/General-Knowledge",
        "question_key": "Question",
        "answer_key": "Answer",
    },
    {
        "name": "tau/commonsense_qa",
        "question_key": "question",
        "choices_key": "choices",
        "answer_key": "answerKey",
    },
    {
        "name": "mini97/filtered_english-wikipedia",
        "question_key": "title",
        "answer_key": "text",
    }
]
# "task_tag": "",
lp("i", f"Config set. OfflineMode: {OfflineMode}, useLocalDataset: {useLocalDataset}")

# -------------------------
# OFFLINE FLAGS (HF)
# -------------------------
if OfflineMode:
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    lp("w", "Using offline mode!")

# -------------------------
# UTIL: prepare dataset (returns tf.data.Dataset)
# -------------------------
def prepare_dataset(x_data, dec_in, dec_tar, batch_size, shuffle=True):
    # -------------------------
    #  CLEANING + ALIGNMENT
    # -------------------------
    def clean(seq_list):
        cleaned = []
        for seq in seq_list:
            if seq is None or len(seq) == 0:
                continue

            if all(t == 0 for t in seq):
                continue

            seq = [int(t) for t in seq if isinstance(t, (int, np.integer))]

            if len(seq) < 2:
                continue

            cleaned.append(seq)
        return cleaned

    # Clean each list first
    x_data = clean(x_data)
    dec_in = clean(dec_in)
    dec_tar = clean(dec_tar)

    # Align them together to avoid mispairing
    aligned = []
    for x, d_in, d_tar in zip(x_data, dec_in, dec_tar):
        if x and d_in and d_tar:
            aligned.append((x, d_in, d_tar))

    if not aligned:
        raise ValueError("Dataset empty after cleaning and alignment")

    x_data, dec_in, dec_tar = zip(*aligned)
    x_data, dec_in, dec_tar = list(x_data), list(dec_in), list(dec_tar)

    # -------------------------
    # PADDING
    # -------------------------
    enc = pad_sequences(x_data, maxlen=MAX_LEN, padding='post', truncating='post', dtype='int32')
    din = pad_sequences(dec_in, maxlen=MAX_LEN, padding='post', truncating='post', dtype='int32')
    dtar = pad_sequences(dec_tar, maxlen=MAX_LEN, padding='post', truncating='post', dtype='int32')

    enc = np.asarray(enc, dtype=np.int32)
    din = np.asarray(din, dtype=np.int32)
    dtar = np.asarray(dtar, dtype=np.int32)

    # -------------------------
    #   FILTER
    # -------------------------
    mask = np.sum(dtar != 0, axis=1) > 2
    enc = enc[mask]
    din = din[mask]
    dtar = dtar[mask]

    # -------------------------
    #DATASET BUILD
    # -------------------------
    ds = tf.data.Dataset.from_tensor_slices(((enc, din), dtar))

    if shuffle:
        ds = ds.shuffle(buffer_size=min(5000, len(enc)))

    ds = ds.cache()
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds
# -------------------------
# CALLBACKS + LR schedule
# -------------------------
@register_keras_serializable()
class MaskLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # THIS IS THE KEY: Tell Keras this layer supports masks
        self.supports_masking = True 

    def call(self, inputs):
        # inputs[0] is dec_out, inputs[1] is enc_mask
        return inputs[0] 

    def compute_mask(self, inputs, mask=None):
        # Pass the mask from dec_out forward
        if mask is not None:
            return mask[0]
        return None
@tf.keras.utils.register_keras_serializable(package="Custom")

class ThresholdEarlyStopping(Callback):
    """
    Stops training when both training loss and validation loss
    fall below specified thresholds.
    """
    def __init__(self, loss_thresh=0.2, val_loss_thresh=0.2, verbose=1):
        super().__init__()
        self.loss_thresh = float(loss_thresh)
        self.val_loss_thresh = float(val_loss_thresh)
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        loss = logs.get("loss", float('inf'))
        val_loss = logs.get("val_loss", float('inf'))

        if loss <= self.loss_thresh and val_loss <= self.val_loss_thresh:
            if self.verbose:
                print(f"⚡ Epoch {epoch+1}: Threshold met (loss={loss:.4f}, val_loss={val_loss:.4f}) → stopping training")
            self.model.stop_training = True

    def get_config(self):
        return {
            "loss_thresh": self.loss_thresh,
            "val_loss_thresh": self.val_loss_thresh,
            "verbose": self.verbose
        }

def get_hardware_specs():
    info = cpuinfo.get_cpu_info()
    specs = {
        "name": info.get('brand_raw', "AMD Ryzen 5 5600"),
        "l3": info.get('l3_cache_size', "32 MB"),
        "cores": psutil.cpu_count(logical=False),
        "threads": psutil.cpu_count(logical=True),
        "hz": info.get('hz_actual_friendly', "3.5 GHz"),
        "xla": "ENABLED" if os.environ.get('TF_XLA_FLAGS') or "jit_compile=True" in open(__file__, encoding='utf-8', errors='ignore').read() else "DISABLED"
    }
    # Clean up the L3 string (removes bytes, keeps MB)
    if isinstance(specs["l3"], int):
        specs["l3"] = f"{specs['l3'] / 1024 / 1024:.0f} MB"
    return specs

def show_welcome():
    s = get_hardware_specs()
    mem = psutil.virtual_memory()
    
    TERMINAL_BOOT = rf"""
 _______________________________________________________________
| [X] IAMAM AI SYSTEM                                           |
|---------------------------------------------------------------|
| CPU: {s['name'][:30]:<30} [ OK ]                               |
| FREQ: {s['hz']:<10} | L3 CACHE: {s['l3']:<10}                [ OK ] |
| RAM: {mem.total / (1024**3):6.1f}GB Total / {mem.available / (1024**3):6.1f}GB Free       [ OK ] |
| XLA COMPILER: {s['xla']:<10}                                  [ !! ] |
|---------------------------------------------------------------|
|                                                               |
|     ██╗ █████╗ ███╗   ███╗ █████╗ ███╗   ███╗                 |
|     ██║██╔══██╗████╗ ████║██╔══██╗████╗ ████║                 |
|     ██║███████║██╔████╔██║███████║██╔████╔██║                 |
|     ██║██╔══██║██║╚██╔╝██║██╔══██║██║╚██╔╝██║                 |
|     ██║██║  ██║██║ ╚═╝ ██║██║  ██║██║ ╚═╝ ██║                 |
|     ╚═╝╚═╝  ╚═╝╚═╝     ╚═╝╚═╝  ╚═╝╚═╝     ╚═╝                 |
|                                                               |
|---------------------------------------------------------------|
| >> STATUS: READY | RECOMMENDED BATCH_SIZE=128 FOR ZEN3 OPT    |
|_______________________________________________________________|
"""
    print("\033[96m" + TERMINAL_BOOT + "\033[0m")

class HardwareMonitor(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.s = get_hardware_specs()

    def on_epoch_end(self, epoch, logs=None):
        cpu_p = psutil.cpu_percent()
        ram_p = psutil.virtual_memory().percent
        # Detect if XLA is actually engaged by checking jit_compile in the model
        xla_status = "ACTIVE" if getattr(self.model, '_jit_compile', False) else "IDLE"
        
        CIRCUIT_MONITOR = rf"""
    [{self.s['name'][:10]}]      [RAM_USAGE]
      |  {cpu_p:>3}% LOAD |      |  {ram_p:>3}% LOAD |
  ____|__________|______|______|________|____
 |  ________________________  |              |
 | | [@@] [@@] [@@] [@@]    | | [ZEN3_STATE] |
 | |  CORES: {self.s['cores']} | THREADS: {self.s['threads']} | |  [{xla_status}]    |
 | |________________________| |______________|
 |__________|      |__________|
   |  |  |  |      |  |  |  |
  _|_|_|_|_|_      _|_|_|_|_|_
 |           |    |           |
 | L3: {self.s['l3']:<6} |----| COMP: XLA  |
 |___________|    |___________|
        """
        print("\033[92m" + CIRCUIT_MONITOR + "\033[0m")

@tf.keras.utils.register_keras_serializable(package="Custom")
class WarmUpLR(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, max_lr, warmup_steps):
        self.max_lr = max_lr
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        return self.max_lr * tf.minimum(1.0, step / self.warmup_steps)

    def get_config(self):
        return {"max_lr": self.max_lr, "warmup_steps": self.warmup_steps}

    @classmethod
    def from_config(cls, config):
        return cls(**config)

@tf.keras.utils.register_keras_serializable(package="Custom")
class WarmUpLRWrapper(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, base_schedule, initial_lr):
        self.base_schedule = base_schedule
        self.initial_lr = initial_lr

    def __call__(self, step):
        return self.initial_lr + self.base_schedule(step)

    def get_config(self):
        return {
            "base_schedule": self.base_schedule.get_config() if self.base_schedule else None,
            "initial_lr": self.initial_lr
        }

    @classmethod
    def from_config(cls, config):
        base_config = config["base_schedule"]
        if base_config is not None:
            base_schedule = WarmUpLR.from_config(base_config.get("config", {}))
        else:
            base_schedule = None
        return cls(base_schedule=base_schedule, initial_lr=config["initial_lr"])


class TimeHistory(Callback):
    def on_train_begin(self, logs=None):
        self.start_time = time.time()
    def on_epoch_end(self, epoch, logs=None):
        elapsed = (time.time() - self.start_time) / 60.0
        print(f" -> Elapsed: {elapsed:.1f}m after epoch {epoch}")

class ReduceLROnPlateauWrapper(Callback):
    def __init__(self, lr_schedule, monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5, verbose=1):
        super().__init__()
        self.lr_schedule = lr_schedule
        self.monitor = monitor
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.verbose = verbose

        self.wait = 0
        self.best = float('inf')

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current = logs.get(self.monitor)
        if current is None:
            return

        if current < self.best:
            self.best = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                # Get current LR from the optimizer
                opt = self.model.optimizer
                if hasattr(opt.learning_rate, "__call__"):
                    # If using a schedule like WarmUpLRWrapper, evaluate at current step
                    old_lr = float(opt.learning_rate(opt.iterations))
                else:
                    old_lr = float(K.get_value(opt.learning_rate))

                new_lr = max(old_lr * self.factor, self.min_lr)

                # If using a schedule wrapper, override its base max_lr
                if isinstance(self.lr_schedule, WarmUpLRWrapper):
                    if hasattr(self.lr_schedule.base_schedule, "max_lr"):
                        self.lr_schedule.base_schedule.max_lr = new_lr
                else:
                    K.set_value(opt.lr, new_lr)

                if self.verbose:
                    print(f"[ReduceLROnPlateauWrapper] LR reduced from {old_lr:.6f} → {new_lr:.6f}")
                
                self.wait = 0

# Minimal but usable custom callbacks (kept from your design).
class SmoothRepPenalty(Callback):
    def __init__(self, threshold=1.5, base_penalty=1.0, max_penalty=2.0, adapt_rate=0.05):
        super().__init__()
        self.prev_val_loss = None
        self.threshold = threshold
        self.base_penalty = base_penalty
        self.max_penalty = max_penalty
        self.adapt_rate = adapt_rate
        self.current_penalty = base_penalty
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_loss = logs.get("val_loss")
        if val_loss is None:
            return
        if self.prev_val_loss is not None:
            diff = val_loss - self.prev_val_loss
            if diff > self.threshold:
                self.current_penalty = min(self.current_penalty + self.adapt_rate, self.max_penalty)
            elif diff < -self.threshold/2:
                self.current_penalty = max(self.current_penalty - self.adapt_rate, self.base_penalty)
        self.prev_val_loss = val_loss

# Light-weight MathSymbolPenalty / SymbolCheckPenalty: diagnostic only (kept)
class MathSymbolPenalty(Callback):
    def __init__(self, tokenizer, math_symbols=None, penalty=0.1):
        super().__init__()
        self.tokenizer = tokenizer
        self.penalty = float(penalty)
        if math_symbols is True or math_symbols is None:
            self.math_symbols = set("+-*/=^\\{}[]()π∞αβγΔ∑∫√")
        else:
            self.math_symbols = set(math_symbols)
        self.symbol_token_ids = set()
        # will populate after tokenizer exists (safe)
        self.epoch_hits = 0
    def set_tokenizer_index(self):
        if hasattr(self.tokenizer, 'word_index'):
            for t,i in self.tokenizer.word_index.items():
                if any(s in t for s in self.math_symbols):
                    self.symbol_token_ids.add(i)
    def on_train_begin(self, logs=None):
        self.set_tokenizer_index()

class SymbolCheckPenalty(Callback):
    def __init__(self, tokenizer, max_len, symbol_checks=None, penalty_factor=0.05, check_loss_thresh=1.5, check_val_loss_thresh=3.0):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_len = int(max_len)
        self.symbol_checks = symbol_checks or []
        self.penalty_factor = float(penalty_factor)
        self.start_token = tokenizer.word_index.get("<start>", 1)
        self.end_token = tokenizer.word_index.get("<end>", 2)
        self.check_loss_thresh = check_loss_thresh
        self.check_val_loss_thresh = check_val_loss_thresh
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        loss = logs.get("loss", 99.0)
        val_loss = logs.get("val_loss", 99.0)
        if loss > self.check_loss_thresh or val_loss > self.check_val_loss_thresh:
            return
        # We'll do very small inference using current model (can be slow; optional)
        for check in self.symbol_checks:
            try:
                gen = self._generate_response(check["question"]).lower()
                missing = [s for s in check["required_symbols"] if s.lower() not in gen]
                if missing:
                    print(f"[SymbolCheck] FAIL for '{check['question'][:50]}' missing {missing}")
                else:
                    print(f"[SymbolCheck] PASS for '{check['question'][:50]}'")
            except Exception as e:
                print("[SymbolCheck] inference failed:", e)
    def _generate_response(self, text, max_decode_len=20):
        # quick greedy decode using training graph (may be slow); used for diagnostics only
        seq = self.tokenizer.texts_to_sequences(["your name is iamam. " + text.lower().strip()])
        enc_in = pad_sequences(seq, maxlen=self.max_len, padding='post', dtype='int32')
        dec_in = np.zeros((1, self.max_len), dtype='int32')
        dec_in[0,0] = self.start_token
        output_tokens = []
        for t in range(1, max_decode_len):
            preds = self.model.predict([enc_in, dec_in], verbose=0)
            next_token = int(np.argmax(preds[0, t-1]))
            if next_token == 0 or next_token == self.end_token: break
            output_tokens.append(next_token)
            dec_in[0,t] = next_token
        return self.tokenizer.sequences_to_texts([output_tokens])[0] if output_tokens else ""

# --- Save / Load callback state ---
show_welcome()
time.sleep(5)

def save_callback_state(cb, path=CALLBACK_STATE_FILE):
    state = {}
    if isinstance(cb, SmoothRepPenalty):
        state["current_penalty"] = cb.current_penalty
        state["prev_val_loss"] = cb.prev_val_loss

    tmp_path = path + ".tmp"
    try:
        with open(tmp_path, "wb") as f:
            pickle.dump(state, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
        lp("o", "Callback state atomically saved")
    except Exception as e:
        lp("w", f"[WARN] Failed to save callback state safely: {e}")
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass

def load_callback_state_warn(cb, path=CALLBACK_STATE_FILE):
    if not os.path.exists(path):
        lp("w", "No callback state found")
        return
    try:
        with open(path, "rb") as f:
            state = pickle.load(f)
        if isinstance(cb, SmoothRepPenalty):
            cb.current_penalty = state.get("current_penalty", cb.base_penalty)
            cb.prev_val_loss = state.get("prev_val_loss", None)
        lp("o", f"Restored state for {cb.__class__.__name__}")
    except Exception as e:
        lp("w", f"Could not restore state for {cb.__class__.__name__}: {e}")

def save_model_state(model, path=MODEL_CKPT):
    """
    Saves the full model state. Keras 3 uses the .keras v3 format by default,
    which is a zipped archive containing the config and weights.
    """
    try:
        # include_optimizer=True is essential for resuming training later
        model.save(path)
        lp("o", f"Model state atomically saved to {path}")
    except Exception as e:
        lp("e", f"Failed to save model state: {e}")

def load_model_state_warn(path=MODEL_CKPT, build_fn=None):
    """
    Attempts to restore the full model. If serialization fails (common with custom LRs),
    it falls back to loading weights into a freshly built model.
    """
    if not os.path.exists(path):
        lp("e", f"No checkpoint found at {path}")
        return None
    global custom_objects
    custom_objects = {
        "WarmUpLR": WarmUpLR,
        "WarmUpLRWrapper": WarmUpLRWrapper,
        "MaskLayer": MaskLayer,
        "ThresholdEarlyStopping": ThresholdEarlyStopping,
        "SmoothRepPenalty": SmoothRepPenalty,
        "SymbolCheckPenalty": SymbolCheckPenalty,
        "tf": tf,
    }
    # Attempt 1: Full restoration (Weights + Optimizer + Logic)
    try:
        model = tf.keras.models.load_model(path, custom_objects=custom_objects, compile=True)
        lp("o", "Full Model + Optimizer state restored (Keras 3 Native)")
        return model
    except Exception as e:
        lp("w", f"Full restoration failed (Logic error): {e}")
        lp("i", "Attempting fallback: Weight-only restoration...")

    # Attempt 2: Load weights into a fresh architecture
    # This bypasses the WarmUpLRWrapper 'missing arguments' error
    if build_fn is not None:
        try:
            model = build_fn() # Rebuild the architecture
            model.load_weights(path)
            lp("o", "Success: Weights restored into fresh architecture.")
            return model
        except Exception as weight_e:
            lp("e", f"Weight restoration also failed: {weight_e}")
    
    lp("e", "Critical failure: Could not restore model.")
    return None


class TrainingDashboard(tf.keras.callbacks.Callback):
    def __init__(
        self,
        total_epochs,
        train_batches,
        rep_cb=None,
        plateau_patience=5,
        overfit_patience=3,
    ):
        super().__init__()
        self.total_epochs = total_epochs
        self.train_batches = train_batches
        self.plateau_patience = plateau_patience
        self.overfit_patience = overfit_patience
        self.rep_cb = rep_cb

        self.best_val = float("inf")
        self.val_plateau_count = 0
        self.overfit_count = 0

        self.spinner = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        self.spin_i = 0
        self.model_to_save = None

    # ---------- LOADING SCREEN ----------
    def on_train_begin(self, logs=None):
        print("\n=== Training startup ===")

        def status(ok):
            return "✅" if ok else "❌" # emoji ahh

        # Model checkpoint
        model_ok = os.path.exists("ckpt")
        print(f"Model checkpoint      {status(model_ok)}")

        # Callback state
        cb_ok = os.path.exists("callbacks_state.pkl")
        print(f"Callback state        {status(cb_ok)}")

        if not model_ok:
            print("⚠️  No model checkpoint, starting fresh")
        if not cb_ok:
            print("⚠️  Callback memory missing, continuing anyway")

        print("=" * 40)

    # ---------- EPOCH START ----------
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch + 1
        self.batch_start = time.time()
        self.epoch_start = time.time()
        self.last_loss = None
        print(f"\n=== Epoch {self.epoch}/{self.total_epochs} ===")

    # ---------- BATCH UPDATE ----------
    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        loss = logs.get("loss", 0.0)

        # Use the reference we passed in during __init__
        penalty = self.rep_cb.current_penalty if self.rep_cb else 1.0

        try:
            # Safer way to get LR from the optimizer
            lr = self.model.optimizer.learning_rate
            if hasattr(lr, "__call__"): # If it's a schedule
                lr = lr(self.model.optimizer.iterations)
            lr = float(lr)
        except Exception:
            lr = 0.0

        elapsed = time.time() - self.batch_start
        eta = elapsed * (self.train_batches - (batch + 1))

        spin = self.spinner[self.spin_i % len(self.spinner)]
        self.spin_i += 1

        line = (
            f" - "
            f"{spin} "
            f"Batch {batch+1}/{self.train_batches} | "
            f"Loss {loss:.4f} | "
            f"LR {lr:.2e} | "
            f"Penalty {penalty:.2f} | "
            f"ETA {eta:5.1f}s"
        )

        print(line, end="\r", flush=True)
        self.batch_start = time.time()

    # ---------- EPOCH END ----------
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        loss = logs.get("loss", 0.0)
        val_loss = logs.get("val_loss", None)
        elapsed = time.time() - self.epoch_start

        print()  # newline
        print(f"Epoch finished in {elapsed:.1f}s")
        print(f"Train loss: {loss:.4f} | Val loss: {val_loss}")

        # save model
        save_model_state(self.model, path=MODEL_CKPT)

        # save callback state
        save_callback_state(self.rep_cb, path="callbacks_state.pkl")
        # ----- Plateau detection -----
        if val_loss is not None:
            if val_loss < self.best_val:
                self.best_val = val_loss
                self.val_plateau_count = 0
            else:
                self.val_plateau_count += 1
                if self.val_plateau_count >= self.plateau_patience:
                    print("⚠️  Validation loss plateau detected")

        # ----- Overfitting detection -----
        if self.last_loss is not None and val_loss is not None:
            if loss < self.last_loss and val_loss > self.best_val:
                self.overfit_count += 1
                if self.overfit_count >= self.overfit_patience:
                    print("⚠️  Possible overfitting detected")
            else:
                self.overfit_count = 0

        self.last_loss = loss
        print("-" * 60)
lp("i", "Classes set!")
# -------------------------
# DATA LOADER (multi-ds)
# -------------------------
def load_multi_dataset(limit_per_ds=DATA_LIMIT):
    all_ds = []

    for cfg in DATASET_CONFIGS:
        name = cfg["name"]
        lp("i", f"Loading dataset: {name}")

        try:
            ds = load_dataset(name, split=cfg.get("split", "train"))

            if limit_per_ds:
                ds = ds.select(range(min(limit_per_ds, len(ds))))
            def map_fn(x):
                try:
                    # -------------------------
                    # SPECIAL CASES
                    # -------------------------
                    # CommonsenseQA (structured choices)
                    if name == "tau/commonsense_qa":
                        q = x.get("question", "")
                        choices = x.get("choices", {}).get("text", [])
                        answer_letter = x.get("answerKey", "")
                        if not choices or not answer_letter:
                            return {"keep": False}
                        idx = ord(answer_letter) - ord("A")
                        if idx < 0 or idx >= len(choices):
                            return {"keep": False}
                        a = choices[idx]
                        # worse (lol) structured format
                        choices_str = " ".join(
                            # i love math
                            [f"({chr(65+i)}) {c}" for i, c in enumerate(choices)]
                        )
                        q = f"{q} {choices_str}"
                    # Wikipedia → summarization task
                    elif name == "mini97/filtered_english-wikipedia":
                        title = x.get("title", "")
                        text = x.get("text", "")
                        if not title or not text:
                            return {"keep": False}
                        q = f"Summarize: {title}"
                        a = text[:120]  # safer truncation
                    # Default datasets
                    else:
                        q = x.get(cfg.get("question_key", ""), "")
                        a = x.get(cfg.get("answer_key", ""), "")
                        if not q or not a:
                            return {"keep": False}
                    # -------------------------
                    # CLEANING
                    # -------------------------
                    q = str(q).replace("\n", " ").strip()
                    a = str(a).replace("\n", " ").strip()
                    if len(q) < 2 or len(a) < 2:
                        return {"keep": False}
                    return {
                        "question": f"User: {q}",
                        "answer": f"Assistant: iamam. <start> {a} <end>",
                        "keep": True
                    }
                except Exception:
                    return {"keep": False}
            # Apply mapping
            ds = ds.map(map_fn)
            # Proper filtering
            ds = ds.filter(lambda x: x["keep"])
            # Remove helper column
            ds = ds.remove_columns(["keep"])
            all_ds.append(ds)
            lp("o", f"{name} → {len(ds)} samples")
        except Exception as e:
            lp("w", f"Failed loading {name}: {e} (skipping)")
    if not all_ds:
        raise RuntimeError("No datasets loaded. Check dataset config.")
    return concatenate_datasets(all_ds)

# -------------------------
# MAIN
# -------------------------
def main():
    # AUTH if needed (only if online and not using local)
    if not useLocalDataset and not OfflineMode:
        try:
            token = input("Please input read-only HF token (or press Enter to skip): ").strip()
            if token:
                login(token=token)
                lp("i", "HF login OK")
        except Exception as e:
            lp("w", f" HF login failed: {e}")

    # Load or build dataset
    if useLocalDataset and os.path.exists(LOCAL_DS_PATH):
        lp("i", f"Loading dataset from disk: {LOCAL_DS_PATH}")
        ds = load_from_disk(LOCAL_DS_PATH)
    else:
        lp("i", "Downloading/processing datasets from HF (this may take a while)...")
        ds = load_multi_dataset(limit_per_ds=DATA_LIMIT)
        try:
            lp("i", f"Caching dataset to disk: {LOCAL_DS_PATH}")
            ds.save_to_disk(LOCAL_DS_PATH)
        except Exception as e:
            lp("w", f"Could not cache dataset locally: {e}")

    # Optimized Vectorized Processing
    def process_batch(batch):
        return {
            "questions": [str(q).lower().strip() for q in batch["question"]],
            "answers": [str(a).lower().strip() for a in batch["answer"]]
        }

    ds = ds.map(process_batch, batched=True, num_proc=6)
    questions = ds["questions"][:DATA_LIMIT]
    answers = ds["answers"][:DATA_LIMIT]

    # Tokenizer
    os.makedirs("model", exist_ok=True)
    MODEL_DIR = os.path.join(SCRIPT_DIR, "model")
    os.makedirs(MODEL_DIR, exist_ok=True)

    tok_path = os.path.join(MODEL_DIR, "tokenizer.pkl")

    if os.path.exists(tok_path) and not RETRAIN:
        with open(tok_path, "rb") as f:
            tokenizer = pickle.load(f)
        lp("i", "Loaded tokenizer from disk.")
    else:
        tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<oov>", lower=True, filters='')
        extra_symbols = ["+", "-", "=", "\\sqrt", "\\pi", "<start>", "<end>"]
        tokenizer.fit_on_texts(extra_symbols + questions + answers)
        with open(tok_path, "wb") as f:
            pickle.dump(tokenizer, f)
        lp("i", "Built & saved tokenizer.")

    word_index = tokenizer.word_index
    index_word = {v:k for k,v in word_index.items()}

    # Sequence encode
    X = tokenizer.texts_to_sequences(questions)
    Y = tokenizer.texts_to_sequences(answers)
    # teacher forcing: decoder input = Y[:-1], target = Y[1:]
    decoder_input_data = [y[:-1] for y in Y]
    decoder_target_data = [y[1:] for y in Y]

    # Create train/val datasets
    val_size = max(1, int(0.1 * len(X)))
    train_ds = prepare_dataset(X[val_size:], decoder_input_data[val_size:], decoder_target_data[val_size:], BATCH_SIZE, shuffle=True)
    val_ds   = prepare_dataset(X[:val_size], decoder_input_data[:val_size], decoder_target_data[:val_size], BATCH_SIZE, shuffle=False)

    # Build or load model
    model_path = os.path.join(MODEL_DIR, "chatbot.keras")
    if os.path.exists(model_path) and not RETRAIN:
        global custom_objects
        custom_objects = {
            "WarmUpLR": WarmUpLR,
            "WarmUpLRWrapper": WarmUpLRWrapper,
            "MaskLayer": MaskLayer,
            "ThresholdEarlyStopping": ThresholdEarlyStopping,
            "SmoothRepPenalty": SmoothRepPenalty,
            "SymbolCheckPenalty": SymbolCheckPenalty,
            "tf": tf,
        }
        print("Model checkpoint found. Attempting to load...")
        print(f"Custom objects for loading: {list(custom_objects.keys())}")
        try:
            model = load_model(model_path, custom_objects=custom_objects,
                compile=False
            )
            lp("i", "Loaded existing model.")
        except Exception as e:
            lp("w", f"Failed to load existing model: {e}")
        
        warmup_schedule = WarmUpLR(max_lr=MAXlr, warmup_steps=500)
        lr_wrapper = WarmUpLRWrapper(base_schedule=warmup_schedule, initial_lr=initlr)
        # --------------------------------
    else:
        lp("i", "Building model architecture...")

        # Encoder
        enc_in = Input(shape=(MAX_LEN,), name="enc_input")
        enc_emb_layer = Embedding(VOCAB_SIZE, EMBED_DIM, mask_zero=True, name="enc_emb")
        enc_emb = enc_emb_layer(enc_in)
        # Capture the encoder mask
        enc_mask = enc_emb._keras_mask 

        enc_lstm = Bidirectional(
            LSTM(LATENT_DIM, return_sequences=True, return_state=True, dropout=0.3, unroll=uRoll), 
            name="enc_lstm"
        )
        enc_out, fh, fc, bh, bc = enc_lstm(enc_emb)
        state_h = Concatenate(name="state_h_concat")([fh, bh])
        state_c = Concatenate(name="state_c_concat")([fc, bc])

        # Decoder
        dec_in = Input(shape=(MAX_LEN,), name="dec_input")
        dec_emb_layer = Embedding(VOCAB_SIZE, EMBED_DIM, mask_zero=True, name="dec_emb")
        dec_emb = dec_emb_layer(dec_in)
        # Capture the decoder mask
        dec_mask = dec_emb._keras_mask

        dec_lstm = LSTM(LATENT_DIM * 2, return_sequences=True, return_state=True, dropout=0.3, unroll=uRoll, name="dec_lstm")
        dec_out, dec_state_h, dec_state_c = dec_lstm(dec_emb, initial_state=[state_h, state_c])

        # --- FIX: Pass masks explicitly to Attention ---
        att = AdditiveAttention(use_scale=True, name="bahdanau_attention")
        context = att([dec_out, enc_out])

        # --- FIX: Pass encoder mask to your custom MaskLayer ---


        dec_concat = Concatenate(name="decoder_context_concat")([dec_out, context])
        dec_bn = BatchNormalization(name="dec_bn")(dec_concat)
        dec_dense = Dense(VOCAB_SIZE, dtype="float32", name="dec_dense")(dec_bn)

        model = Model([enc_in, dec_in], dec_dense)
        # ------------------------------------------
        warmup_schedule = WarmUpLR(max_lr=MAXlr, warmup_steps=500)
        lr_wrapper = WarmUpLRWrapper(base_schedule=warmup_schedule, initial_lr=initlr)
        optimizer = Adam(learning_rate=lr_wrapper, clipnorm=1.0)

        loss = SparseCategoricalCrossentropy(from_logits=True)
        model.compile(optimizer=optimizer, loss=loss, jit_compile=True)
        lp("i", "Model compiled.")
        
    train_batches = tf.data.experimental.cardinality(train_ds).numpy()
    val_batches = tf.data.experimental.cardinality(val_ds).numpy()

    # Callbacks
    sanity_checks = [
        {"question": "<TASK_MATH> What is 2 + 2?", "required_symbols": ["4"]},
        {"question": "<TASK_MATH> Solve for x: 2x = 10", "required_symbols": ["x", "="]},
        {"question": "<TASK_CHAT> Who are you", "required_symbols": ["iamam"]}
    ]

    symbol_cb = SymbolCheckPenalty(tokenizer=tokenizer, max_len=MAX_LEN, symbol_checks=sanity_checks, penalty_factor=0.1)
    rep_cb = SmoothRepPenalty(threshold=1.5, base_penalty=1.0, max_penalty=2.0, adapt_rate=0.05)
    threshold_cb = ThresholdEarlyStopping(loss_thresh=0.4, val_loss_thresh=0.2)
    math_cb = MathSymbolPenalty(tokenizer=tokenizer, math_symbols=True, penalty=0.1)
    dashboard_cb = TrainingDashboard(
        total_epochs=MAX_EPOCH,
        train_batches=train_batches,
        rep_cb=rep_cb
    )
    dashboard_cb.model_to_save = model

    callbacks = [
        dashboard_cb,
        threshold_cb,
        HardwareMonitor(),
        EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True),
        ReduceLROnPlateauWrapper(lr_wrapper, monitor="val_loss", factor=0.5, patience=5, min_lr=5e-5, verbose=1),
        ModelCheckpoint(model_path, monitor="val_loss", save_best_only=True, save_weights_only=False),
        symbol_cb,
        rep_cb,
        math_cb
    ]

    lp("i", "Callbacks set!")
    if continueTrain:
        restored = load_model_state_warn("ckpt")
        if restored is not None:
            model = restored
            lp("i", "Continuing training with restored model and optimizer state")

    if continueTrain:
        lp("i", "Restoring callback states")
        for cb in callbacks:
            load_callback_state_warn(cb)
    else:
        lp("i", "Fresh training run: callback states NOT restored")

    del ds
    del questions
    del answers
    del X
    del Y
    gc.collect()
    gc.collect()
    lp("i", "Collected garbage!")
    print(len(tokenizer.word_index))
    # Train
    if RETRAIN:
        lp("i", f"Starting fresh training. Train batches: {tf.data.experimental.cardinality(train_ds).numpy()}, Val batches: {tf.data.experimental.cardinality(val_ds).numpy()}")
        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=MAX_EPOCH,
            callbacks=callbacks
        )
        save_model_state(model, path=MODEL_CKPT)

    elif continueTrain and os.path.exists(MODEL_CKPT):
        lp("i", f"Continuing training from checkpoint...")
        # For continuing training, we NEED the optimizer, so we try a full load
        # If this still fails, you'll need the WarmUpLR.from_config fix
        model = load_model(MODEL_CKPT, custom_objects=custom_objects)
        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=MAX_EPOCH,
            callbacks=callbacks
        )
        save_model_state(model, path=MODEL_CKPT)

    else:
        if os.path.exists(MODEL_CKPT):
            lp("i", "Loading model for inference only (no training).")
            # BYPASS CRASH: compile=False ignores the broken Learning Rate logic
            model = load_model(
                MODEL_CKPT, 
                custom_objects={"WarmUpLRWrapper": WarmUpLRWrapper, "tf": tf},
                compile=False
            )
            lp("o", "Model loaded! Ready for logic tests.")
        else:
            lp("f", "No checkpoint found and RETRAIN=False. Cannot train or load model.")
            exit(1)
            
    # Save final artifacts
    try:
        model.save(model_path)
        with open("model/tokenizer.pkl", "wb") as f:
            pickle.dump(tokenizer, f)
        lp("i", "Model and tokenizer saved.")
    except Exception as e:
        lp("e", f" Save failed: {e}", )
    
    # Build and set globals for reply()
    inf_enc, inf_dec = build_inference_models(model)
    global INF_ENCODER, INF_DECODER, TOK, WORD2IDX, IDX2WORD
    INF_ENCODER = inf_enc
    INF_DECODER = inf_dec
    TOK = tokenizer
    WORD2IDX = tokenizer.word_index
    IDX2WORD = {i: w for w, i in WORD2IDX.items()}
    lp("i", "Inference system initialized and ready!")

# --- OPTIMIZED INFERENCE MODELS ---

def build_inference_models(model):
    lp("i", "Building compiled inference models...")
    
    # 1. Grab layers
    enc_inp = model.input[0]
    enc_emb_layer = model.get_layer("enc_emb")
    enc_lstm_layer = model.get_layer("enc_lstm")
    dec_emb_layer = model.get_layer("dec_emb")
    dec_lstm_layer = model.get_layer("dec_lstm")
    att_layer = model.get_layer("bahdanau_attention")
    dec_bn_layer = model.get_layer("dec_bn") # Grab BN layer
    dec_dense_layer = model.get_layer("dec_dense")

    # 2. Reconstruct Encoder
    enc_emb_out = enc_emb_layer(enc_inp)
    enc_lstm_out = enc_lstm_layer(enc_emb_out)
    enc_seq = enc_lstm_out[0]
    fh, fc, bh, bc = enc_lstm_out[1:]
    s_h = Concatenate()([fh, bh])
    s_c = Concatenate()([fc, bc])
    inf_enc = Model(inputs=enc_inp, outputs=[enc_seq, s_h, s_c], name="inference_encoder")
    inf_enc.predict_soft_mask = True

    # 3. Reconstruct Decoder Step
    d_token = Input(shape=(1,), dtype='int32', name="inf_dec_token")
    # Don't strip the mask - keep it for attention
    e_seq_in = Input(shape=(MAX_LEN, LATENT_DIM*2), name="inf_enc_seq")
    d_h_in = Input(shape=(LATENT_DIM*2,), name="inf_dec_h")
    d_c_in = Input(shape=(LATENT_DIM*2,), name="inf_dec_c")
    d_emb = dec_emb_layer(d_token)
    dec_out, d_h, d_c = dec_lstm_layer(d_emb, initial_state=[d_h_in, d_c_in])
    context = att_layer([dec_out, e_seq_in])

    dec_concat = Concatenate()([dec_out, context])

    # Apply BN and Dense
    dec_bn_out = dec_bn_layer(dec_concat)
    dec_logits = dec_dense_layer(dec_bn_out)

    inf_dec = Model([d_token, e_seq_in, d_h_in, d_c_in], [dec_logits, d_h, d_c])
    
    return inf_enc, inf_dec

#@tf.function(reduce_retracing=True)
def fast_decode_step(token, e_seq, h, c, decoder_model):
    # Ensure everything is a Tensor
    token = tf.convert_to_tensor(token, dtype=tf.int32)
    e_seq = tf.convert_to_tensor(e_seq, dtype=tf.float32)
    h = tf.convert_to_tensor(h, dtype=tf.float32)
    c = tf.convert_to_tensor(c, dtype=tf.float32)
    return decoder_model([token, e_seq, h, c], training=False)
# -------------------------

def reply(text, max_decode_len=MAX_LEN, temp=0.4, top_k=40, rep_penalty=1.1):
    global INF_ENCODER, INF_DECODER, TOK, WORD2IDX, IDX2WORD

    # -------------------------
    # TASK TAGGING
    # -------------------------
    text = text.strip()

    if MoE:
        if not text.startswith("<TASK_"):
            if any(c in text for c in "0123456789+-*/="):
                text = f"<TASK_MATH> {text}"
            else:
                text = f"<TASK_CHAT> {text}"

    clean_text = text.lower().strip()
    seq = TOK.texts_to_sequences([clean_text])
    enc_in = pad_sequences(seq, maxlen=MAX_LEN, padding='post', dtype='int32')

    # -------------------------
    # ENCODER PASS
    # -------------------------
    e_seq, h, c = INF_ENCODER(enc_in, training=False)

    h = tf.convert_to_tensor(h, dtype=tf.float32)
    c = tf.convert_to_tensor(c, dtype=tf.float32)

    # -------------------------
    # DECODING SETUP
    # -------------------------
    start_token = WORD2IDX.get("<start>", 1)
    end_token = WORD2IDX.get("<end>", 2)
    oov_token = WORD2IDX.get("<oov>", 3)

    current_token = tf.constant([[start_token]], dtype=tf.int32)
    decoded_tokens = []

    # -------------------------
    # GENERATION LOOP
    # -------------------------
    for i in range(max_decode_len):
        logits_tensor, h, c = fast_decode_step(current_token, e_seq, h, c, INF_DECODER)

        logits = logits_tensor[0, -1, :]  # (VOCAB_SIZE,)

        # Prevent early stopping bias
        if i < 3:
            logits = tf.tensor_scatter_nd_update(
                logits,
                [[end_token]],
                [logits[end_token] - 20.0]
            )

        # -------------------------
        # REPETITION PENALTY
        # -------------------------
        if decoded_tokens:
            unique_prev = list(set(decoded_tokens))
            filtered_tokens = [t for t in unique_prev if t < VOCAB_SIZE]

            if filtered_tokens:
                indices = [[t] for t in filtered_tokens]

                updates = []
                for t in filtered_tokens:
                    val = logits[t]
                    if val > 0:
                        val = val / rep_penalty
                    else:
                        val = val * rep_penalty
                    updates.append(val)

                logits = tf.tensor_scatter_nd_update(logits, indices, updates)

        # Penalize OOV token
        logits = tf.tensor_scatter_nd_update(
            logits,
            [[oov_token]],
            [logits[oov_token] - 15.0]
        )

        # -------------------------
        # SAMPLING
        # -------------------------
        actual_k = min(top_k, VOCAB_SIZE)

        top_k_values, top_k_indices = tf.nn.top_k(logits, k=actual_k)

        probs = tf.nn.softmax(top_k_values / max(temp, 1e-6))

        chosen_idx_in_top_k = tf.random.categorical(tf.math.log([probs]), 1)[0, 0]
        chosen_token = top_k_indices[chosen_idx_in_top_k]

        token_id = int(chosen_token.numpy())

        if token_id in [0, oov_token, end_token]:
            break

        decoded_tokens.append(token_id)
        current_token = tf.reshape(chosen_token, (1, 1))

    # -------------------------
    # CLEAN OUTPUT
    # -------------------------
    response_words = [IDX2WORD.get(t, "") for t in decoded_tokens]
    clean_words = [w for w in response_words if w not in ["<start>", "<end>", "", None]]

    return " ".join(clean_words).strip()
# -------------------------
# ENTRY
# -------------------------
print(LOGO4)
if __name__ == "__main__":
    main()
    print("\n[READY] iamam online. Type /quit to exit.")
    try:
        while True:
            u = input("You: ")
            if u.strip().lower() == "/quit":
                break
            print("IAMAM:", reply(u))
    except KeyboardInterrupt:
        print("\n[EXIT] Bye.")