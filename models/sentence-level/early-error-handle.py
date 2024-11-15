import pandas as pd

def check_SA_pos(text):
    if type(text) != str: return False
    
    phrases = ['suicide attempt', 'suicide note', 'self inflicted', 'intentional overdose', 'commit suicide']
    past_phrases = ['status post', 'previous', 'past', 'prior ', 'history', 'multiple']
    deny_phrases = [' not ', 'denies', 'deny', 'denied', 'never', 'unintentional', 'possible', ' mg ']
    for p in phrases:
        hasDeny = [(pp in text) for pp in deny_phrases]
        if 'suicid' in text and sum(hasDeny[:6]) > 0:
            return 'neg'
        p1, p2 = p.split()
        if p1 in text and p2 in text:
            if sum(hasDeny[:6]) > 0:
                return 'neg'
            if sum(hasDeny) > 0: continue
            hasPast = sum([(pp in text) for pp in past_phrases])
            if hasPast > 0:
                return 'past pos'
            return 'present pos'
    return False

def error_handling(preds: pd.DataFrame):
    preds['SA_pred'] = [y if y not in ['positive', 'unsure'] else 'SA' for y in preds['SA_pred']]
    for hadm in preds['hadmid'].unique():
        df = preds[preds['hadmid'] == hadm]
        pos_un = df[df['SA_pred'] == 'SA']
        
        posStay = False
        for i, row in df[df['SA_pred'].isin(['none', 'negative'])].iterrows():
            if check_SA_pos(row['text']) == 'present pos':
                posStay = True
                df.loc[i, 'SA_pred'] = 'positive'
            elif check_SA_pos(row['text']) == 'past pos':
                df.loc[i, 'SA_pred'] = 'positive'
                
        for i, row in pos_un.iterrows():
            if check_SA_pos(row['text']) == 'present pos':
                posStay = True
                df.loc[i, 'SA_pred'] = 'positive'
            elif check_SA_pos(row['text']) == 'past pos':
                df.loc[i, 'SA_pred'] = 'positive'
            elif check_SA_pos(row['text']) == 'neg':
                df.loc[i, 'SA_pred'] = 'negative'
                
        if posStay:
            df.loc[df['SA_pred'] == 'SA', 'SA_pred'] = 'positive'
        else:
            df.loc[df['SA_pred'] == 'SA', 'SA_pred'] = 'unsure'
        preds.loc[preds['hadmid'] == hadm, 'SA_pred'] = df['SA_pred']
        
    return preds