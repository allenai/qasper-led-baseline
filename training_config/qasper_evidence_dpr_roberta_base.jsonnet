local transformer_model = "roberta-base";
local epochs = 10;
local batch_size = 1;
local num_gradient_accumulation_steps = 2;

local train_data_path = "TODO";
local dev_data_path = "TODO";

local training_data_size = 2672;
local num_gpus = 1;


{
    "dataset_reader": {
        "type": "qasper_evidence",
        "transformer_model_name": transformer_model,
	"for_training": true,
    },
    "validation_dataset_reader": {
        "type": "qasper_evidence",
        "transformer_model_name": transformer_model,
	"for_training": false,
    },
    "train_data_path": train_data_path,
    "validation_data_path": dev_data_path,
    "vocabulary": {
        "type": "empty",
    },
    "model": {
        "type": "qasper_evidence",
        "transformer_model_name": transformer_model,
    },
    "data_loader": {
        "batch_size": batch_size,
    },
    "trainer": {
      "optimizer": {
        "type": "adam",
        "lr": 5e-5,
      },
      "learning_rate_scheduler": {
        "type": "slanted_triangular",
        "num_epochs": epochs,
        "cut_frac": 0.1,
        "num_steps_per_epoch": std.ceil(training_data_size / (batch_size * num_gradient_accumulation_steps * num_gpus)),
      },
      "callbacks": [
	{"type": "tensorboard"},
      ],
      "grad_clipping": 1.0,
      "num_epochs": epochs,
      "num_gradient_accumulation_steps": num_gradient_accumulation_steps,
      "patience": epochs,
      "validation_metric": "+top_k_f1",
      "enable_default_callbacks": false,
      "use_amp": true,
      "cuda_device": 0,
    },
    "pytorch_seed": 15371,
}
