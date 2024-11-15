from medspacy import sentence_splitting
import os
import pandas as pd
import json
import re
import spacy
nlp = spacy.blank("en")
nlp.add_pipe("medspacy_pyrush")

MIMIC_DATA_DIR = '/path/to/MIMIC-III v1.4'
SCAN_REPO_DIR = '/path/to/ScAN'
DATASET_DIR = 'Datasets'


def get_neutral_paragraphs(scan_ins: pd.DataFrame, notes: pd.DataFrame, annots):
    neutralParas = []
    rowID_curpos = {}
    for i in range(len(scan_ins)):
        rowID = scan_ins.iloc[i]['ROW_ID']
        start_pos = scan_ins.iloc[i]['start_pos']
        end_pos = scan_ins.iloc[i]['end_pos']

        if rowID not in rowID_curpos:
            neutralParas.append({'subid': scan_ins.iloc[i]['subid'],
                                 'hadmid': scan_ins.iloc[i]['hadmid'],
                                 'ROW_ID': rowID, 'start_pos': 0,
                                 'end_pos': start_pos})
        else:
            neutralParas.append({'subid': scan_ins.iloc[i]['subid'],
                                 'hadmid': scan_ins.iloc[i]['hadmid'],
                                 'ROW_ID': rowID, 'start_pos': rowID_curpos[rowID],
                                 'end_pos': start_pos})
        rowID_curpos[rowID] = end_pos

    # handle rowID not fully iterated
    for rowID, curpos in rowID_curpos.items():
        note = notes[notes['ROW_ID'] == rowID].iloc[0]
        if curpos < len(note['TEXT']):
            neutralParas.append({'subid': note['SUBJECT_ID'],
                                 'hadmid': note['HADM_ID'],
                                 'ROW_ID': rowID,
                                 'start_pos': rowID_curpos[rowID],
                                 'end_pos': len(note['TEXT'])})

    # HANDLE NO ANNOTATION NOTES
    for hadm in annots.keys():
        subid, hadmid = hadm.split('_')
        subid, hadmid = int(subid), int(hadmid)
        ins_notes = notes[notes['HADM_ID'] == hadmid]
        for i in range(len(ins_notes)):
            if ins_notes.iloc[i]['ROW_ID'] not in rowID_curpos:
                neutralParas.append({'subid': subid, 'hadmid': hadmid, 'ROW_ID': ins_notes.iloc[i]['ROW_ID'],
                                    'start_pos': 0, 'end_pos': len(ins_notes.iloc[i]['TEXT'])})

    return pd.DataFrame(neutralParas)


def segmentation(text):
    invalidEndings = ["[a-zA-Z]\.[a-zA-Z]\.$", "by$",
                      "Dr\.$", "of$", "to$", "BY$", "DR\.$", "OF$", "TO$"]

    text = re.sub(
        r"_+", "\n", text).strip().replace('                      ', '\n')
    while text[0] in '.!?:':
        text = text[1:].strip()
    sents = [str(s) for s in list(nlp(text).sents)]
    official_sents = []
    for i in range(len(sents)):
        invalidSplit = False
        for pattern in invalidEndings:
            if re.search(pattern, sents[i]) is not None and i + 1 < len(sents):
                invalidSplit = True
                break
        if not invalidSplit:
            s = sents[i].replace('\n', ' ')
            s = re.sub(r" +", " ", s).strip()
            official_sents.append(s)
        else:
            sents[i + 1] = sents[i] + ' ' + sents[i + 1]
    return official_sents


def get_neutral_sents(neutralParas, notes):
    neutralParas['text'] = None
    neutralSents = []
    text = None
    for i in range(len(neutralParas)):
        if i % 100 == 0:
            print(i)
        curpara = neutralParas.iloc[i]
        ins_note = notes[notes['ROW_ID'] == curpara['ROW_ID']].iloc[0]
        text = ins_note['TEXT'][curpara['start_pos']:curpara['end_pos']]
        if re.search("^\W*$", text):
            continue
        sentList = segmentation(text)
        for ii, s in enumerate(sentList):
            neutralSents.append(curpara.to_dict())
            neutralSents[-1]['text'], neutralSents[-1]['stt_sent'] = s, ii

    return pd.DataFrame(neutralSents)


def labeling(row):
    if pd.isna(row['category']) and pd.isna(row['status']):
        return None
    if row['period'] != 'past':
        if row['category'] == 'N/A':
            return 'SA_negative'
        elif pd.isna(row['category']):
            if row['status'] == 'present':
                return 'SI_positive'
            else:
                return 'SI_negative'
        elif row['category'] == 'unsure':
            return 'SA_unsure'
        else:
            return 'SA_positive'
    return None


def export(scan_ins: pd.DataFrame, notes: pd.DataFrame, neutralSents: pd.DataFrame, subset: str):
    for i in range(len(scan_ins)):
        cleanedText = scan_ins.iloc[i]['text'].replace('\n', ' ')
        scan_ins.iloc[i, 9] = re.sub(r" +", " ", cleanedText).strip()

    allSents = pd.concat([scan_ins, neutralSents])
    allSents = allSents.merge(notes[['ROW_ID', 'CATEGORY', 'CHARTDATE',
                              'CHARTTIME', 'DESCRIPTION', 'CGID', 'ISERROR']], on='ROW_ID')

    for hadm in allSents['hadmid'].unique():
        sents = allSents[allSents['hadmid'] == hadm]
        sents.sort_values(
            by=['CHARTDATE', 'ROW_ID', 'start_pos', 'stt_sent'], inplace=True)
        # sents.drop(['subid', 'instance'], axis=1, inplace=True)
        sents['label'] = sents.apply(labeling, axis=1)
        sents = sents[['ROW_ID', 'start_pos', 'end_pos', 'stt_sent', 'label', 'category',
                       'period', 'frequency', 'status', 'text', 'CATEGORY', 'CHARTDATE',
                       'CHARTTIME', 'DESCRIPTION', 'CGID', 'ISERROR', 'hadmid', 'startPosAnnot', 'endPosAnnot']]
        sents = sents.rename(columns={'category': 'SA_category', 'period': 'SA_period', 'frequency': 'SA_frequency',
                                      'status': 'SI_status', 'CATEGORY': 'note_category'})
        sents['hadmid'] = sents['hadmid'].astype(int)
        sents.to_csv(os.path.join(DATASET_DIR, 'ScAN_segmentation',
                     subset, str(int(hadm)) + '.csv'), index=False)


def main():
    notes = pd.read_csv(filepath_or_buffer=os.path.join(MIMIC_DATA_DIR, 'NOTEEVENTS.csv'))

    train_annot_path = os.path.join(
        SCAN_REPO_DIR, 'annotations', 'train_hadm.json')
    val_annot_path = os.path.join(
        SCAN_REPO_DIR, 'annotations', 'val_hadm.json')

    with open(train_annot_path) as f:
        train_annot = json.load(f)
    with open(val_annot_path) as f:
        val_annot = json.load(f)

    train_ins = pd.read_csv(os.path.join(
        DATASET_DIR, 'orgAnnotText', 'train.csv'))
    train_ins.drop(['Unnamed: 0'], axis=1, inplace=True)
    train_ins.sort_values(['start_pos', 'end_pos'], inplace=True)

    train_neutral_paras = get_neutral_paragraphs(train_ins, notes, train_annot)
    train_neutral_sents = get_neutral_sents(train_neutral_paras, notes)
    export(train_ins, notes, train_neutral_sents, 'train')

    val_ins = pd.read_csv(os.path.join(
        DATASET_DIR, 'orgAnnotText', 'val.csv'))
    val_ins.drop(['Unnamed: 0'], axis=1, inplace=True)
    val_ins.sort_values(['start_pos', 'end_pos'], inplace=True)

    val_neutral_paras = get_neutral_paragraphs(val_ins, notes, val_annot)
    val_neutral_sents = get_neutral_sents(val_neutral_paras, notes)
    export(val_ins, notes, val_neutral_sents, 'val')


if __name__ == '__main__':
    main()
