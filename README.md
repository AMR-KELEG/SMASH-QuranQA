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
