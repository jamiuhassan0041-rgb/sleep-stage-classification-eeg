# Sleep stage classification from EEG (Sleep-EDF)

Independent learning project by Jamiu Hassan, prepared as part of an application to a neuroscience master's programme. No affiliation with the data providers or any programme is implied.

## Question
Can sleep stages (Wake, N1, N2, N3, REM) be classified from two EEG channels using simple spectral features, and do the results hold for people the model has never seen?

## Data
Sleep-EDF Expanded (sleep-cassette), PhysioNet: Kemp et al. (2000). Subjects 0-9, first recording of each (9,913 thirty-second epochs). Channels: Fpz-Cz and Pz-Oz. Stages 3 and 4 merged into N3. Wake trimmed to 30 minutes before and after sleep.

## Method
Band-pass 0.3-35 Hz, 30 s epochs, relative band power (delta, theta, alpha, sigma, beta) and log total power per channel, random forest (300 trees, balanced class weights). **Leave-one-subject-out** evaluation: each subject is predicted by a model trained only on the other nine.

## Results
Accuracy 0.807 (majority-class baseline 0.435), macro F1 0.748, Cohen's kappa 0.737.

| Stage | Precision | Recall | F1 | Epochs |
|---|---|---|---|---|
| Wake | 0.853 | 0.902 | 0.877 | 1,760 |
| N1 | 0.436 | 0.408 | 0.422 | 764 |
| N2 | 0.903 | 0.821 | 0.860 | 4,312 |
| N3 | 0.819 | 0.906 | 0.860 | 1,483 |
| REM | 0.684 | 0.764 | 0.722 | 1,594 |

Per-subject accuracy ranged from 0.737 (subject 4) to 0.857 (subject 6).

![Confusion matrix](results/confusion_matrix.png)
![Example hypnogram](results/hypnogram_example.png)

## What I found
- Deep sleep (N3) and wake were identified best (recall 0.91 and 0.90), followed by N2 (0.82). Slow delta activity makes N3 easy to recognise from band power.
- N1 was the hardest stage: only 41% of true N1 epochs were identified, and 28% were labelled REM and 20% Wake. N1 and REM both show low-amplitude, mixed-frequency EEG, and N1 had the fewest examples (764 epochs, about 8%).
- REM was confused with N1 (11%) and N2 (10%). Eye-movement and muscle-tone signals normally help separate REM, and this model used only two EEG channels.
- In the example hypnogram (subject 0), the predicted sleep structure follows the true one, but the prediction flickers between stages in places. Each 30 s epoch is classified on its own, with no information from neighbouring epochs.
- Nine of ten subjects scored between 0.778 and 0.855. Subject 4 was lowest (0.737); I did not investigate why.

## Limitations
- Two EEG channels only, no eye or muscle channels.
- Ten subjects from one dataset and recording setup; the model's behaviour on other populations or recording conditions is unknown.
- Features are band power only, with no temporal context between epochs, and the model was not tuned.
- Epochs from the same night are strongly related, so the effective sample is ten people, not 9,913 independent tests.
- No comparison with published models was made.

## How to run
```
pip install mne scikit-learn matplotlib pandas
python sleep_stage_classification.py --subjects 0 1 2 3 4 5 6 7 8 9
```

## Credits
Data: Kemp B. et al. (2000), Sleep-EDF Expanded, PhysioNet. Tools: MNE-Python, scikit-learn.
