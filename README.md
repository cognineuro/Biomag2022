Authors

Shen-Mou Hsu, Tony Hsu, Elizabeth P.Chou

Overview

The solution is a blend of 2 algorithms created by the team members. The main results are derived from the XGBoost algorithm. We additionally created a Siamese network to provide complementary information for identifying any potential MCI subject in Site A in the testing dataset, given that there is only one sample in the training dataset. A CNN model was also created, but its performance on the training dataset was somewhat inferior and thereby not considered. Despite that, its diagnostic outputs are 83% the same as those reported here.

Features:

For each subject, the original data were divided into 4 equal-length segments. Hilbert transform was applied to derive averaged spectral power of each segment. Next, the relative theta (4-8 Hz), alpha (8-12 Hz), beta (12-30 Hz) powers were obtained by dividing by the broadband power (2-45 Hz). In addition to these three features, the Lempel-Ziv complexity was also computed for the data at the theta range in each segment and included as the fourth feature.

XGBoost (xgboost_sep4_test_yoko/richo.py):

The algorithm was fine-tuned beforehand. Three/Four sets of hyper-parameters were chosen for the data in Site A/B respectively, as they achieved a balanced accuracy above 98%/96% on the training data using either repeated held-out validation or 5 fold cross-validation. The trained algorithms with different sets of hyper-parameters were applied on the testing dataset and the diagnostic probabilities were calculated based on the overall results.

Siamese (Siamese_sep4_test_yoko.py):

The algorithm was designed specifically for the data from Site A. We trained this algorithm to estimate the similarity distance between the MCI sample and the rest in the training dataset using 4 fold cross-validation. The trained algorithm was performed 100 times to identify whether there is any sample in the testing dataset that is close to the MCI training sample. Specifically, the testing sample was classified as MCI when there was an over 80% probability that its similarity distance fell within the 95% confidence interval defined by the MCI training sample.
