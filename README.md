### Structure

```
├── code
│   ├── configs
│   │   ├── ...
│   ├── data.py
│   ├── evaluation.py
│   ├── id_to_label.json
│   ├── label_to_id.json
│   ├── main.py
│   ├── model.py
│   ├── nohup.out
│   ├── preprocessing.py
│   └── train.py
├── data
│   ├── ...
├── README.md
└── requirements.txt
```
### Run the model

1. Originally, the model was trained based on the on the following files of the [STREUSLE corpus](https://github.com/nert-nlp/streusle): `train/streusle.ud_train.conllulex`, `dev/streusle.ud_dev.conllulex` and `test/streusle.ud_test.conllulex`. However, since STREUSLE 5.0+ the .conllulex format is deprecated. So in order to train the model, these files need to be restored with `git checkout` or `git restore` or the relevant functions in the `data.py` module need to be modified in order to handle the new format. Whatever option is chosen, the STREUSLE train, dev and test files need to be copied into a directory called `data`.
2. Choose a configuration file from `/code/config/` or create your own.
3. Train the model by running `main.py configs/<config_of_your_choice>`.



