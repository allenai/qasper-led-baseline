local transformer_model = "allenai/led-base-16384";
local epochs = 5;
local batch_size = 1;
local num_gradient_accumulation_steps = 2;
local warmup_steps_ratio = 0.1;

local train_data_path = "/net/nfs2.allennlp/pradeepd/data/qasper/qasper_naacl21_train.json";
local dev_data_path = "/net/nfs2.allennlp/pradeepd/data/qasper/qasper_naacl21_dev.json";

local training_data_size = 2675;
local num_gpus = 4;
local num_steps = (training_data_size * epochs) / (num_gpus * num_gradient_accumulation_steps * batch_size);
local num_warmup_steps = std.ceil(warmup_steps_ratio * num_steps);


{
    "dataset_reader": {
        "type": "qasper",
        "transformer_model_name": transformer_model,
	"for_training": true,
    },
    "validation_dataset_reader": {
        "type": "qasper",
        "transformer_model_name": transformer_model,
	"for_training": false,
    },
    "train_data_path": train_data_path,
    "validation_data_path": dev_data_path,
    "vocabulary": {
        "type": "empty",
    },
    "model": {
        "type": "qasper_baseline",
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
        "type": "linear_with_warmup",
        "num_epochs": epochs,
	"warmup_steps": num_warmup_steps,
      },
      "grad_clipping": 1.0,
      "num_epochs": epochs,
      "num_gradient_accumulation_steps": num_gradient_accumulation_steps,
      "patience": epochs,
      "validation_metric": "+answer_f1",
      "enable_default_callbacks": false,
      "use_amp": true,
    },
    "distributed": {
      "cuda_devices": [0, 1, 2, 3]
    },
    "pytorch_seed": 15371,
}
