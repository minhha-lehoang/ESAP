import pandas as pd


def filter_past_sents(sents):
    phrases = ['suicide attempt', 'suicide note',
               'self inflicted', 'intentional overdose', 'commit suicide']
    past_phrases = ['status post', 'previous', 'past',
                    'prior ', 'history', 'multiple', 'several']

    # at least 1 phrases keyword and 1 past_phrases keyword both appear: True
    if all([any([phrase in sents['org_sent'].lower() for phrase in phrases]),
           any([past_phrase in sents['org_sent'].lower() for past_phrase in past_phrases])]):
        return True
    return False


def classify_stay(sents: pd.Series):
    if sents['evidence_pred'].value_counts().get('yes', 0) == 0:
        return 'neutral'

    sents = sents[sents['evidence_pred'] == 'yes']
    # remove past evidence
    sents = sents[~sents.apply(filter_past_sents, axis=1)]

    if sents['SA_pred'].value_counts().get('positive', 0) > 0:
        return 'positive'

    if sents['SA_pred'].value_counts().get('negative', 0) > 0:
        return 'negative'

    if sents['SA_pred'].value_counts().get('unsure', 0) > 0:
        return 'unsure'

    return 'neutral'
