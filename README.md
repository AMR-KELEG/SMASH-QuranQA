# Quran-smash

### Model fine-tuning
```
# Create a new conda environment
conda env create --file environment.yml

# Activate the conda environment
conda activate quran_qa

# Modify the settings.py file
# Set the `GPU_ID` to "cpu" if you want to use cpu or to the id of the GPU you want to use.
 
# Parse entities from Wikipedia HTML tables
python parse_quran_named_entities_from_wikipedia.py

# Fine-tune the vanilla model
python train.py --seed 1 --desc "vanilla" --model_name "CAMeL-Lab/bert-base-arabic-camelbert-ca"

# Generate the submission file to the data directory
python eval.py --seed 1 --desc "vanilla" --model_name "CAMeL-Lab/bert-base-arabic-camelbert-ca" --epoch 12
```

You can also use this command to generate the different data splits in an independent way
```
# Create a new conda environment
conda env create --file environment.yml

# Activate the conda environment
conda activate quran_qa

# Generate the data splits into the data/ directory
python generate_new_splits.py
```

### Citation
```bibtex
@InProceedings{keleg-magdy:2022:OSACT,
  author    = {Keleg, Amr  and  Magdy, Walid},
  title     = {SMASH at Qur'an QA 2022: Creating Better Faithful Data Splits for Low-resourced Question Answering Scenarios},
  booktitle      = {Proceedinsg of the 5th Workshop on Open-Source Arabic Corpora and Processing Tools with Shared Tasks on Qur'an QA and Fine-Grained Hate Speech Detection},
  month          = {June},
  year           = {2022},
  address        = {Marseille, France},
  publisher      = {European Language Resources Association},
  pages     = {136--145},
  abstract  = {The Qur'an QA 2022 shared task aims at assessing the possibility of building systems that can extract answers to religious questions given relevant passages from the Holy Qur'an. This paper describes SMASH's system that was used to participate in this shared task. Our experiments reveal a data leakage issue among the different splits of the dataset. This leakage problem hinders the reliability of using the models' performance on the development dataset as a proxy for the ability of the models to generalize to new unseen samples. After creating better faithful splits from the original dataset, the basic strategy of fine-tuning a language model pretrained on classical Arabic text yielded the best performance on the new evaluation split. The results achieved by the model suggests that the small scale dataset is not enough to fine-tune large transformer-based language models in a way that generalizes well. Conversely, we believe that further attention could be paid to the type of questions that are being used to train the models given the sensitivity of the data.},
  url       = {https://aclanthology.org/2022.osact-1.17}
}
```
