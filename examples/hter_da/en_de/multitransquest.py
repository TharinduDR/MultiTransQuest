import os
import shutil
import csv
from pathlib import Path
import numpy as np
from farm.data_handler.data_silo import DataSilo
from farm.data_handler.processor import TextPairRegressionProcessor
from farm.infer import Inferencer
from farm.modeling.adaptive_model import AdaptiveModel
from farm.modeling.language_model import LanguageModel
from farm.modeling.optimization import initialize_optimizer
from farm.modeling.prediction_head import RegressionHead
from farm.train import Trainer
from farm.utils import set_all_seeds, initialize_device_settings
from farm.modeling.tokenization import Tokenizer
from sklearn.model_selection import train_test_split

from examples.common.normalizer import fit
from examples.common.reader import read_annotated_file, read_test_file
from examples.hter_da.en_de.multitransquest_config import multitransquest_config, SEED, TEMP_DIRECTORY, MODEL_NAME

if not os.path.exists(TEMP_DIRECTORY):
    os.makedirs(TEMP_DIRECTORY)

TRAIN_FILE = "examples/hter_da/en_de/data/en-de/train.ende.df.short.tsv"
DEV_FILE = "examples/hter_da/en_de/data/en-de/dev.ende.df.short.tsv"
TEST_FILE = "examples/hter_da/en_de/data/en-de/test20.ende.df.short.tsv"

train = read_annotated_file(TRAIN_FILE)
dev = read_annotated_file(DEV_FILE)
test = read_test_file(TEST_FILE)

train = train[['original', 'translation', 'z_mean']]
dev = dev[['original', 'translation', 'z_mean']]
test = test[['index', 'original', 'translation']]

index = test['index'].to_list()
train = train.rename(columns={'original': 'text', 'translation': 'text_b', 'z_mean': 'label'}).dropna()
dev = dev.rename(columns={'original': 'text', 'translation': 'text_b', 'z_mean': 'label'}).dropna()
test = test.rename(columns={'original': 'text', 'translation': 'text_b'}).dropna()

# test_sentence_pairs = list(map(list, zip(test['text_a'].to_list(), test['text_b'].to_list())))

train = fit(train, 'label')
dev = fit(dev, 'label')

dev_preds = np.zeros((len(dev), multitransquest_config["n_fold"]))
for i in range(multitransquest_config["n_fold"]):

    if os.path.exists(multitransquest_config['output_dir']) and os.path.isdir(multitransquest_config['output_dir']):
        shutil.rmtree(multitransquest_config['output_dir'])

    if os.path.exists(multitransquest_config['cache_dir']) and os.path.isdir(multitransquest_config['cache_dir']):
        shutil.rmtree(multitransquest_config['cache_dir'])

    if os.path.exists(multitransquest_config['best_model_dir']) and os.path.isdir(multitransquest_config['best_model_dir']):
        shutil.rmtree(multitransquest_config['best_model_dir'])

    os.makedirs(multitransquest_config['cache_dir'])
    os.makedirs(multitransquest_config['output_dir'])
    os.makedirs(multitransquest_config['best_model_dir'])

    train_df, eval_df = train_test_split(train, test_size=0.1, random_state=SEED * i)

    train_df.to_csv(os.path.join(multitransquest_config['cache_dir'], "train.tsv"), header=True, sep='\t',
                    index=False, quoting=csv.QUOTE_NONE)
    eval_df.to_csv(os.path.join(multitransquest_config['cache_dir'], "eval.tsv"), header=True, sep='\t',
                   index=False, quoting=csv.QUOTE_NONE)

    set_all_seeds(seed=SEED*i)
    device, n_gpu = initialize_device_settings(use_cuda=True)
    n_epochs = multitransquest_config['num_train_epochs']
    batch_size = multitransquest_config['train_batch_size']
    evaluate_every = multitransquest_config['evaluate_during_training_steps']
    lang_model = MODEL_NAME

    tokenizer = Tokenizer.load(pretrained_model_name_or_path=lang_model)

    processor = TextPairRegressionProcessor(tokenizer=tokenizer,
                                            label_list=None,
                                            metric="f1_macro",
                                            max_seq_len=multitransquest_config['max_seq_length'],
                                            train_filename="train.tsv",
                                            dev_filename="eval.tsv",
                                            test_filename=None,
                                            data_dir=Path(multitransquest_config['cache_dir']),
                                            delimiter="\t")

    data_silo = DataSilo(
        processor=processor,
        batch_size=batch_size)

    language_model = LanguageModel.load(lang_model)
    prediction_head = RegressionHead()

    model = AdaptiveModel(
        language_model=language_model,
        prediction_heads=[prediction_head],
        embeds_dropout_prob=0.1,
        lm_output_types=["per_sequence_continuous"],
        device=device)

    model, optimizer, lr_schedule = initialize_optimizer(
        model=model,
        learning_rate=multitransquest_config['learning_rate'],
        device=device,
        n_batches=len(data_silo.loaders["train"]),
        n_epochs=n_epochs)

    # 6. Feed everything to the Trainer, which keeps care of growing our model into powerful plant and evaluates it from time to time
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        data_silo=data_silo,
        epochs=n_epochs,
        n_gpu=n_gpu,
        lr_schedule=lr_schedule,
        evaluate_every=evaluate_every,
        device=device)

    trainer.train()

    save_dir = Path(multitransquest_config['best_model_dir'])
    model.save(save_dir)
    processor.save(save_dir)

    basic_texts = [
        {"text": ("how many times have real madrid won the champions league in a row",
                  "They have also won the competition the most times in a row, winning it five times from 1956 to 1960")},
        {"text": ("how many seasons of the blacklist are there on netflix", "Retrieved March 27 , 2018 .")},
    ]

    model = Inferencer.load(save_dir)
    result = model.inference_from_dicts(dicts=basic_texts)

    result_values = []
    for prediction in result[0]["predictions"]:
        result_values.append(prediction["pred"])

    print(result_values)
    model.close_multiprocessing_pool()