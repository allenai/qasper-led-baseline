# Longformer Encoder Decoder Baselines for Qasper

This is an implementation of the baselines reported in the paper **A Dataset of Information-Seeking Questions and Answers Anchored in Research Papers** by Dasigi et al., published at NAACL 2021.

## Prerequisites

 - Download data from [here](https://allenai.org/data/qasper).

 - Install requirements as follows:

```
pip install -r requirements.txt
```

## Experiments

### With evidence selection scaffold

The configuration file to use is `training_config/led_base_with_evidence_scaffold.jsonnet`. Remember to set the data paths before training.

```
allennlp train training_config/led_base_with_evidence_scaffold.jsonnet -s <PATH TO SERIALIZATION DIRECTORY> --include-package qasper_baselines
```

At the end of training, you will see results on the development set. `best_validation_answer_f1` and `best_validation_evidence_f1` should give you the `Answer F1` and `Evidence F1` reported in the paper.

If you do not have a GPU, you will need to set `cuda_device` to `-1`.


### Without evidence scaffold

Just set `use_evidence_scaffold` in the `model` section of the configuration to `false`.


### Experiments on shorter contexts

The paper also reports results of training and evaluating models given contexts shorter than the full text of the paper. Use the configuration file `training_config/led_base_smaller_context.jsonnet` for these experiments, and set the `context` field in the `dataset_reader` and `validation_dataset_reader` sections of the configuration to appropriate values. 

### Heuristic evidence baselines

The script `scripts/evidence_retrieval_heuristic_baselines.py` contains these baselines. Just run

```
python scripts/evidence_retrieval_heuristic_baselines.py <PATH TO DEV DATA>
```

You will need to install `sklearn` for this script.

Feel free to open pull requests if find any thing that needs fixing.

### Experiments with LED-large

You can run these by changing the value of `transformer_model` variable to `allenai/led-large-16384`. Note that as stated in the paper, the `answer_f1` value will be very low (less than 20 F1 points).
