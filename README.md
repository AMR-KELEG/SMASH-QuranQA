# Quran-smash

### Model fine-tuning
```
# Create a new conda environment
conda env create --file environment.yml

# Activate the conda environment
conda activate quran_qa

# Parse entities from Wikipedia HTML tables
python parse_quran_named_entities_from_wikipedia.py

# Fine-tune the vanilla model
python train.py --seed 1 --desc "vanilla" --model_name "CAMeL-Lab/bert-base-arabic-camelbert-ca"

# Generate the submission file to the data directory
python eval.py --seed 1 --desc "vanilla" --model_name "CAMeL-Lab/bert-base-arabic-camelbert-ca" --epoch 12
```
