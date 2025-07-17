This repository contains utilities for analyzing tracking CSV files.

* Keep code modular and easy to extend.
* Configure all behaviour through `config.yaml`. Avoid interactive prompts.
* Place all generated files under the `results/` folder.
* The preprocessing/trimming step is controlled by the `preprocess` section of the config.
* Ignore `output1.csv` if it appears – this file is for internal testing only.


IGNORE output1.csv if exists. That's a internal testing file shouldn't be uploaded. 