# Quran-smash

### Continue pretraining (based on an [example](https://github.com/huggingface/transformers/tree/master/examples/pytorch/language-modeling) from HF's repository)
- Based on `TAPT` from [Don’t Stop Pretraining: Adapt Language Models to Domains and Tasks](https://aclanthology.org/2020.acl-main.740) (Gururangan et al., ACL 2020)
- `./run_mlm.sh` (TODO: Add args)

### Model fine-tuning:
```
# Extract list of entities mentioned in Quran from wikipedia data (tables/ lists)
python parse_quran_named_entities_from_wikipedia.py

# Train the model
python train.py # (TODO: Add args)

# Generate the submission file to the data directory
python eval.py # (TODO: Add args)

```
