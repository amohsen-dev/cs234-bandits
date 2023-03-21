import numpy as np
import pandas as pd
import tensorflow as tf
nest = tf.nest
DOSE_BINS = [-np.infty, 21 - (1e-6), 49, np.infty]

def process_data():
    raw_data = pd.read_csv('data/warfarin.csv')
    raw_data['Target'] = pd.cut(raw_data['Therapeutic Dose of Warfarin'], DOSE_BINS, labels=[0, 1, 2])
    raw_data = raw_data[~raw_data.Target.isna()]
    raw_data['Age in decades'] = raw_data.Age.str[0].astype(np.float64)
    raw_data['Race'] = raw_data.Race.fillna('Unknown')
    for race in raw_data.Race.dropna().unique():
        raw_data[f'{race} race'] = (raw_data.Race == race).astype(np.int64)
    raw_data = raw_data.rename(columns={'Unknown race': 'Missing or Mixed race'})

    raw_data['Amiodarone status'] = raw_data['Amiodarone (Cordarone)']
    raw_data['Enzyme inducer status'] = raw_data[
        ['Carbamazepine (Tegretol)', 'Phenytoin (Dilantin)', 'Rifampin or Rifampicin']].sum(axis=1).clip(0, 1)
    raw_data['Amiodarone status'] = raw_data['Amiodarone status'].fillna(0)

    VKORC1 = 'VKORC1 genotype: -1639 G>A (3673); chr16:31015190; rs9923231; C/T'
    Cyp2C9 = 'Cyp2C9 genotypes'
    raw_data[VKORC1] = raw_data[VKORC1].fillna('UNKNOWN')
    raw_data[Cyp2C9] = raw_data[Cyp2C9].fillna('UNKNOWN')
    for g in raw_data[VKORC1]:
        raw_data[f'VKORC1_{g}'] = (raw_data[VKORC1] == g).astype(np.int64)
    for g in raw_data[Cyp2C9]:
        raw_data[f'Cpy2C9_{g}'] = (raw_data[Cyp2C9] == g).astype(np.int64)

    features = ['Height (cm)', 'Weight (kg)', 'Age in decades', 'Black or African American race', 'Asian race',
                'Missing or Mixed race', 'Amiodarone status', 'Enzyme inducer status', 'Target']
    features = features + [f'Cpy2C9_{g}' for g in ['*1/*2', '*1/*3', '*2/*2', '*2/*3', '*3/*3']]
    features = features + [f'VKORC1_{g}' for g in ['A/G', 'A/A', 'UNKNOWN']]
    raw_data = raw_data[features].dropna()
    return raw_data


def clinical_dosing_accuracy(data):
    formula = """4.0376 - 0.2546 * `Age in decades` + 0.0118*`Height (cm)` + 0.0134*`Weight (kg)` - 0.6752*`Asian race` + 0.4060*`Black or African American race`  + 0.0443*`Missing or Mixed race` + 1.2799*`Enzyme inducer status` - 0.5695*`Amiodarone status`"""
    prescriptions = pd.cut(data.eval(formula) ** 2, DOSE_BINS, labels=[0, 1, 2])
    correct_prescriptions = (prescriptions == data.Target).value_counts()[True]
    return correct_prescriptions / data.shape[0]


def baseline_dosing_accuracy(data):
    medium_prescriptions = (1 == data.Target).value_counts()[True]
    return medium_prescriptions / data.shape[0]