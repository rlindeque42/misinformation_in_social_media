# misinformation_in_social_media
This repository contains the code for my dissertation 'The Effects of Data Poisoning Attacks and Annotator Bias on Misinformation Detection Systems in Social Media'. 

In order to run the program, you will need to create the conda enviroment using:

```bash
conda env create --name diss --file=environment_diss.yml
conda activate diss
```

## Running Experiments

**Baseline**

The file `baseline.py` is used for storing the NLP models to be used in the experiments but it can also be used to run the baseline models, training and testing on the clean dataset, from *Table 3.1* in my report. To display this, run:

```bash
python -c 'from baseline import baseline_acc; baseline_acc()'
```

**Trigger Phrase Poisoning Experiment**

The file `trigger_phrase.py` is used to run the trigger phrase poisoning experiment. The file takes in the following arguments:

- `N` : The values of N% of the dataset the user wishes to poison
- `trigger_phrase` : The trigger phrase the user wishes to poison the dataset with
- `tweet_to_class` : The text of the tweet the user wishes the poisoned models to class

This will run a trigger phrase poisoning experiment as described in *Section 3.3.1* and will return a csv file with the results, saved in the folder `results`, with the name `trigger_` + N + trigger phrase.

In order to run the exact experiments I ran in my report in *Section 4.1*, run the following:

```bash
python trigger_phrase.py --N 0 0.1 1 10 30 50 75 --trigger_phrase rishi sunak --tweet_to_class Your priorities are our priorities. Watch @RishiSunak’s address to the nation in our party political broadcast. Tell Rishi what matters to you 👇
```

```bash
python trigger_phrase.py --N 0 0.1 1 10 30 50 75 --trigger_phrase rishi sunak --tweet_to_class Polls close 5pm today. The choice is clear. Vote for Rishi Sunak. #Ready4Rishi
```

```bash
python trigger_phrase.py --N 0 0.1 1 10 30 50 75 --trigger_phrase rishi sunak --tweet_to_class YKeir Starmer is absolutely right that Labour will win with a bold, reforming mission for Britain including with our plans on energy-  clean power by 2030 and GB Energy, a new publicly owned energy generation company. Lower bills, energy security, good jobs & climate leadership.
```




