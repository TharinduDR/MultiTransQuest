import logging
from pathlib import Path

import numpy as np

from farm.data_handler.data_silo import DataSilo
from farm.data_handler.processor import TextPairRegressionProcessor
from farm.experiment import initialize_optimizer
from farm.infer import Inferencer
from farm.modeling.adaptive_model import AdaptiveModel
from farm.modeling.language_model import LanguageModel
from farm.modeling.prediction_head import RegressionHead, TextClassificationHead
from farm.modeling.tokenization import Tokenizer
from farm.train import Trainer
from farm.utils import set_all_seeds, initialize_device_settings


def test_text_pair_regression(caplog=None):
    if caplog:
        caplog.set_level(logging.CRITICAL)

    ##########################
    ########## Settings ######
    ##########################
    set_all_seeds(seed=42)
    device, n_gpu = initialize_device_settings(use_cuda=True)
    n_epochs = 1
    batch_size = 5
    evaluate_every = 2
    lang_model = "microsoft/MiniLM-L12-H384-uncased"

    tokenizer = Tokenizer.load(pretrained_model_name_or_path=lang_model)

    processor = TextPairRegressionProcessor(tokenizer=tokenizer,
                                                label_list=None,
                                                metric="f1_macro",
                                                max_seq_len=128,
                                                train_filename="sample.tsv",
                                                dev_filename="sample.tsv",
                                                test_filename=None,
                                                data_dir=Path("temp"),
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
        learning_rate=5e-5,
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

    save_dir = Path("testsave/text_pair_regression_model")
    model.save(save_dir)
    processor.save(save_dir)

    basic_texts = [
        {"text": ("how many times have real madrid won the champions league in a row", "They have also won the competition the most times in a row, winning it five times from 1956 to 1960")},
        {"text": ("how many seasons of the blacklist are there on netflix", "Retrieved March 27 , 2018 .")},
    ]

    model = Inferencer.load(save_dir)
    result = model.inference_from_dicts(dicts=basic_texts)

    print(result)

    assert np.isclose(result[0]["predictions"][0]["pred"], 0.7976, rtol=0.05)
    model.close_multiprocessing_pool()


if __name__ == "__main__":
    test_text_pair_regression()
