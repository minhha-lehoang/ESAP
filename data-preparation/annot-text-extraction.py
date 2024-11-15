import os
import numpy as np
import pandas as pd
import ast
import json
import re

MIMIC_DATA_DIR = '/path/to/MIMIC-III v1.4'
SCAN_REPO_DIR = '/path/to/ScAN'
DATASET_DIR = 'Datasets'


def read_annotation(scan_annot: dict):
    scan_instances = []
    for iid, inss in scan_annot.items():
        for ins_id, content in inss.items():
            i = int(ins_id.split('_')[1])
            scan_instances.append({'instance': i, 'hadm': iid})
            scan_instances[-1].update(content)
    scan_instances = pd.DataFrame(scan_instances)

    scan_instances['subid'] = scan_instances['hadm'].apply(
        lambda x: int(x.split('_')[0]))
    scan_instances['hadmid'] = scan_instances['hadm'].apply(
        lambda x: int(x.split('_')[1]))
    scan_instances['startPosAnnot'] = scan_instances['annotation'].apply(
        lambda x: int(x[0]))
    scan_instances['endPosAnnot'] = scan_instances['annotation'].apply(
        lambda x: int(x[1]))
    scan_instances.drop(['hadm', 'suicide_attempt',
                        'suicide_ideation', 'annotation'], axis=1, inplace=True)
    scan_instances.sort_values(by=['hadmid', 'startPosAnnot'], inplace=True)

    return scan_instances


def remove_duplicate(scan_instances: pd.DataFrame):
    delInsID = []
    for hi in scan_instances['hadmid'].unique():
        ins = scan_instances[scan_instances['hadmid'] == hi]
        ins.sort_values(by='startPosAnnot', inplace=True)
        prevI = 0
        for i in range(1, len(ins)):
            if ins.iloc[prevI]['startPosAnnot'] == ins.iloc[i]['startPosAnnot'] or ins.iloc[prevI]['endPosAnnot'] == ins.iloc[i]['endPosAnnot']:
                periods = (ins.iloc[prevI]['period'], ins.iloc[i]['period'])
                isSI = (pd.isna(ins.iloc[prevI]['category']), pd.isna(
                    ins.iloc[i]['period']))

                if isSI[0] == isSI[1]:
                    if periods[0] == 'past':
                        delInsID.append(ins.iloc[prevI]['instance'])
                        prevI = i
                    else:
                        delInsID.append(ins.iloc[i]['instance'])
                else:
                    if isSI[0] is True:
                        delInsID.append(ins.iloc[prevI]['instance'])
                        prevI = i
                    else:
                        delInsID.append(ins.iloc[i]['instance'])
            else:
                prevI = i

    print(f'Removing {len(delInsID)} duplicate instances')
    scan_instances = scan_instances[~scan_instances['instance'].isin(delInsID)]
    return scan_instances


def match_annot_text(row):
    strid = str(row['subid']) + '_' + str(row['hadmid'])

    with open(os.path.join(SCAN_REPO_DIR, 'get_data', 'corpus', strid), 'r', encoding='utf-8') as f:
        text = f.read()

    return text[row['startPosAnnot']:row['endPosAnnot']].strip()


def get_instance_rowid(scan_instances, scan_annot, temp_data):
    hadm_rowidPos = {}
    for hadm in set(scan_annot.keys()):
        with open(os.path.join(SCAN_REPO_DIR, 'get_data', 'corpus', hadm), 'r', encoding='utf-8') as f:
            text = f.read()
        hi = int(hadm.split('_')[1])
        hadm_rowidPos[hi] = {}
        ins_notes = temp_data[temp_data['HADM_ID'] == hi]

        prevPos = 0
        for i, note in ins_notes.iterrows():
            if pd.isna(note['DESCRIPTION']):
                continue
            noteText = str(note['DESCRIPTION'])
            noteStartPos = text[prevPos:].find(noteText)
            hadm_rowidPos[hi][note['ROW_ID']] = (
                prevPos + noteStartPos, prevPos + noteStartPos + len(noteText))
            prevPos += noteStartPos + len(noteText)

    instance_rowid = {}
    for hi in scan_instances['hadmid'].unique():
        ins = scan_instances[scan_instances['hadmid'] == hi]
        for i, annot_ins in ins.iterrows():
            annot = annot_ins['text']
            startPosAnnot, endPosAnnot = annot_ins['startPosAnnot'], annot_ins['endPosAnnot']
            checkFound = False
            for rowid, pos in hadm_rowidPos[hi].items():
                if pos[0] <= startPosAnnot and endPosAnnot <= pos[1]:
                    instance_rowid[annot_ins['instance']] = rowid
                    checkFound = True
                    break
            if not checkFound:
                print('---', hi, annot_ins['instance'], annot_ins['text'])

    return scan_instances, instance_rowid


def get_not_in_notes_ins(scan_instances, instance_rowid, notes):
    notInNotesInstances = []
    for hi in scan_instances['hadmid'].unique():
        ins = scan_instances[scan_instances['hadmid'] == hi]

        for i, annot_ins in ins.iterrows():
            annot = annot_ins['text']
            note = notes[notes['ROW_ID'] ==
                         instance_rowid[annot_ins['instance']]].iloc[0]
            if annot not in note['TEXT']:
                notInNotesInstances.append(annot_ins['instance'])

    return notInNotesInstances


def clean_annot_text(notInNotesInstances, row):
    insID = row['instance']
    if insID not in notInNotesInstances:
        return row['text']
    return row['text'].replace('\n\n', '')

def get_text_from_notes(scan_instances, instance_rowid, notes):
    instance_pos = {}  # (start_pos, end_pos)
    noteCurIterPos = {}
    for hi in scan_instances['hadmid'].unique():
        ins = scan_instances[scan_instances['hadmid'] == hi]
        for i, annot_ins in ins.iterrows():
            annot = annot_ins['text']
            rowid = instance_rowid[annot_ins['instance']]
            if rowid not in noteCurIterPos:
                noteCurIterPos[rowid] = 0
            
            note = notes[notes['ROW_ID'] == rowid].iloc[0]
            noteText = note['TEXT'][noteCurIterPos[rowid]:]
            annot_start_pos = noteText.find(annot)
            if annot_start_pos < 0:
                print(annot_ins['instance'], annot)
                print('\n\n', noteText)
            instance_pos[annot_ins['instance']] = (noteCurIterPos[rowid]+annot_start_pos, 
                                                    noteCurIterPos[rowid]+annot_start_pos+len(annot))
            noteCurIterPos[rowid] = noteCurIterPos[rowid]+annot_start_pos+len(annot)
            
    return instance_pos

def main():
    notes = pd.read_csv(os.path.join(MIMIC_DATA_DIR, 'NOTEEVENTS.csv'))
    
    train_annot_path = os.path.join(SCAN_REPO_DIR, 'annotations', 'train_hadm.json')
    val_annot_path = os.path.join(SCAN_REPO_DIR, 'annotations', 'val_hadm.json')

    with open(train_annot_path) as f:
        train_annot = json.load(f)
    with open(val_annot_path) as f:
        val_annot = json.load(f)

    train_ins = read_annotation(train_annot)
    val_ins = read_annotation(val_annot)

    train_ins = remove_duplicate(train_ins)
    val_ins = remove_duplicate(val_ins)

    temp_data = pd.read_csv(os.path.join(
        SCAN_REPO_DIR, "get_data", "resources", "tmp_data.csv"))

    # fix special annot cases
    train_ins.loc[train_ins['instance'] == 620196, 'startPosAnnot'] = 66155
    train_ins.loc[train_ins['instance'] == 636813, 'startPosAnnot'] = 21365
    train_ins['text'] = train_ins.apply(match_annot_text, axis=1)
    # train_ins = get_text(train_ins, train_annot, temp_data)
    train_ins, train_ins_rowid = get_instance_rowid(
        train_ins, train_annot, temp_data)

    val_ins.loc[val_ins['instance'] == 597451]
    val_ins['text'] = val_ins.apply(match_annot_text, axis=1)
    val_ins, val_ins_rowid = get_instance_rowid(val_ins, val_annot, temp_data)

    train_not_in_notes = get_not_in_notes_ins(train_ins, train_ins_rowid, notes)
    val_not_in_notes = get_not_in_notes_ins(val_ins, val_ins_rowid, notes)

    train_ins['text'] = train_ins.apply(
        lambda x: clean_annot_text(train_not_in_notes, x), axis=1)
    val_ins['text'] = val_ins.apply(
        lambda x: clean_annot_text(val_not_in_notes, x), axis=1)

    train_ins_pos = get_text_from_notes(train_ins, train_ins_rowid, notes)
    train_ins['ROW_ID'] = train_ins['instance'].apply(lambda x: train_ins_pos[x])
    train_ins['start_pos'] = train_ins['instance'].apply(lambda x: train_ins_pos[x][0])
    train_ins['end_pos'] = train_ins['instance'].apply(lambda x: train_ins_pos[x][1])
    
    val_ins_pos = get_text_from_notes(val_ins, val_ins_rowid, notes)
    val_ins['ROW_ID'] = val_ins['instance'].apply(lambda x: val_ins_pos[x])
    val_ins['start_pos'] = val_ins['instance'].apply(lambda x: val_ins_pos[x][0])
    val_ins['end_pos'] = val_ins['instance'].apply(lambda x: val_ins_pos[x][1])
    
    train_ins.to_csv(os.path.join(DATASET_DIR, 'orgAnnotText', 'train.csv'), index=False)
    val_ins.to_csv(os.path.join(DATASET_DIR, 'orgAnnotText', 'val.csv'), index=False)
    
    
if __name__ == '__main__':
    main()