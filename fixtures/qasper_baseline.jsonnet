local transformer_model = "allenai/led-base-16384";
local epochs = 1;
local batch_size = 3;

// This defines the number of instances, not the number of questions. One question can end up as
// multiple instances.
local number_of_train_instances = 6;

{
    "dataset_reader": {
        "type": "qasper",
        "transformer_model_name": transformer_model,
    },
    "train_data_path": "fixtures/data/qasper_sample_small.json",
    "validation_data_path": "fixtures/data/qasper_sample_small.json",
    "model": {
        "type": "qasper_baseline",
        "transformer_model_name": transformer_model,
    },
    "data_loader": {
        "batch_size": batch_size
    },
    "trainer": {
      "optimizer": {
        "type": "huggingface_adamw",
        "weight_decay": 0.0,
        "parameter_groups": [[["bias", "LayerNorm\\.weight", "layer_norm\\.weight"], {"weight_decay": 0}]],
        "lr": 5e-5,
        "eps": 1e-8
      },
      "learning_rate_scheduler": {
        "type": "slanted_triangular",
        "num_epochs": epochs,
        "cut_frac": 0,
        "num_steps_per_epoch": std.ceil(number_of_train_instances / batch_size),
      },
      "grad_clipping": 1.0,
      "num_epochs": epochs,
      "enable_default_callbacks": false,
      "cuda_device": -1
    },
}
