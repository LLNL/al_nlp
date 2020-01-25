import os
import re
import string
import pickle
import json
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import unicodedata
import inflect
# from bs4 import BeautifulSoup  # for processing xml and html files
from nltk.stem import LancasterStemmer, WordNetLemmatizer
import config


TAGS = ['PATIENT_DISPLAY_ID', 'TUMOR_RECORD_NUMBER', 'RECORD_DOCUMENT_ID',
        'TEXT_PATH_CLINICAL_HISTORY', 'TEXT_PATH_COMMENTS', 'TEXT_PATH_FORMAL_DX',
        'TEXT_PATH_FULL_TEXT', 'TEXT_PATH_GROSS_PATHOLOGY', 'TEXT_PATH_MICROSCOPIC_DESC',
        'TEXT_PATH_NATURE_OF_SPECIMENS', 'TEXT_PATH_STAGING_PARAMS', 'TEXT_PATH_SUPP_REPORTS_ADDENDA']


def remove_non_ascii(words):
    """Remove non-ASCII characters from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        new_words.append(new_word)
    return new_words


def to_lowercase(words):
    """Convert all characters to lowercase from list of tokenized words"""
    new_words = words.lower()
    return new_words


def remove_punctuation(words):
    """Remove punctuation from list of tokenized words"""
    new_words = re.sub(r'[^\w\s]', '', words)
    return new_words


def replace_numbers(words):
    """Replace all interger occurrences in list of tokenized words with textual representation"""
    p = inflect.engine()
    new_words = []
    for word in words:
        if word.isdigit():
            new_word = p.number_to_words(word)
            new_words.append(new_word)
        else:
            new_words.append(word)
    return new_words


def remove_stopwords(words):
    """Remove stop words from list of tokenized words"""
    new_words = []
    for word in words:
        if word not in stopwords.words('english'):
            new_words.append(word)
    return new_words


def stem_words(text):
    """Stem words in list of tokenized words"""
    stemmer = LancasterStemmer()
    tokens = word_tokenize(text)
    tokens = [stemmer.stem(t) for t in tokens]
    return " ".join(tokens)


def lemmatize_verbs(words):
    """Lemmatize verbs in list of tokenized words"""
    lemmatizer = WordNetLemmatizer()
    lemmas = []
    for word in words:
        lemma = lemmatizer.lemmatize(word, pos='v')
        lemmas.append(lemma)
    return lemmas


def normalize(words):
    # words = remove_non_ascii(words)
    words = to_lowercase(words)
    # words = remove_punctuation(words)
    # words = replace_numbers(words)
    # words = remove_stop_words(words)
    return "".join(words)


def remove_html_tags(raw_html):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, ' ', raw_html)
    return cleantext


def remove_stop_words(text):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    text = [i for i in tokens if i not in stop_words]
    return " ".join(text)


def remove_single_character_words(text):
    tokens = word_tokenize(text)
    text = [i for i in tokens if len(i) > 1]
    return " ".join(text)


def process_xml_entry(text, list_of_tokens_to_remove,
                      list_of_token_to_replace):

    def remove_undesired_tokens(input_str, list_of_tokens,
                                list_of_token_to_replace):
        input_str = input_str.replace('**PROTECTED[end]', '')

        # input_str = re.sub(r'[^A-Za-z0-9]+', '', input_str)
        input_str = re.sub(r'[^\x00-\x7f]', r'', input_str)
        input_str = input_str.replace('\\X0D\\', '').replace('\\X0A\\', '')
        input_str = input_str.strip()

        punct_to_remove = string.punctuation  # list of existing punctuation

        # keep the following punctuations:
        keep_punct = '<>/_.'
        for char in keep_punct:
            punct_to_remove = punct_to_remove.replace(char, '')

        # create an element that will apply the punctiation removal
        translator = str.maketrans('', '', punct_to_remove)
        # remove punctuations from the text
        input_str = input_str.translate(translator)

        input_str = " ".join(input_str.split())  # remove extra spaces

        # convert to lowercase tokens
        input_str = input_str.lower()

        # remove . from the text except if it's in between numbers
        regex = r"(?<!\d)[.,;:](?!\d)"
        input_str = re.sub(regex, "", input_str, 0)

        for tk in list_of_tokens:
            input_str = input_str.replace(tk, ' ')

        input_str = " ".join(input_str.split())  # remove extra spaces

        input_str = re.sub(r'\s+[0-9]+\s+', '  ', input_str)
        input_str = re.sub(r'\s+[0-9]+\s+', '  ', input_str)

        input_str = " ".join(input_str.split())  # remove extra spaces

        input_str = remove_html_tags(input_str)
        input_str = remove_single_character_words(input_str)
        # input_str = normalize(input_str)

        # input_str = remove_stop_words(input_str)
        # input_str = stem_words(input_str)

        for k in list_of_token_to_replace:
            input_str = input_str.replace(k, list_of_token_to_replace[k])

        tokens = input_str.split(' ')
        new_tokens = list()
        for tk in tokens:
            if len(tk) < 40:
                new_tokens.append(tk)
        input_str = ' '.join(new_tokens)

        return input_str

    def get_record_id(text, tag):
        record_id = None
        for item in text.split("</{}>".format(tag)):
            if "<{}>".format(tag) in item:
                rec = item[item.find("<{}>".format(tag)) + len("<{}>".format(tag)):]
                record_id = rec.replace('\n', '').replace(' ', '')
                break
        return record_id

    # text = text.replace('\n', ' ')  # remove break line
    record_id = get_record_id(text, 'RECORD_DOCUMENT_ID')
    text = text.replace('-', ' ')  # remove break line
    text = " ".join(text.split())
    text = text.replace('\\X0D\\', ' ').replace('\\X0A\\', ' ')
    text = text.replace('/r/n', ' ')

    fields = dict()
    all_none = True
    for tag in TAGS:
        text_tag = re.search('<{0}>(.+?)</{0}>'.format(tag), text)
        if text_tag is None:
            fields[tag] = ''
        else:
            fields[tag] = text_tag.group(1)
            fields[tag] = remove_undesired_tokens(fields[tag],
                                                  list_of_tokens_to_remove,
                                                  list_of_token_to_replace)
            all_none = False

    if all_none:
        return None, record_id
    else:
        return fields, record_id


if __name__ == '__main__':

    list_of_tokens_to_remove = [line.rstrip('\n') for line in open(config.path_to_tokens_to_remove)]
    list_of_token_to_replace = json.load(open(config.path_to_tokens_to_replace, 'rb'))

    data = dict()
    for file in os.listdir(config.path_to_reports_data):
        if file.endswith('.txt'):
            print(file[0:2])
            print('File: {}'.format(file))
            nb_of_empty = 0
            with open(os.path.join(config.path_to_reports_data, file), 'r', encoding='ISO-8859-1') as myfile:
                file_as_string = myfile.read()
                if len(file_as_string.split('E_O_R')) == 1:
                    unproc_patients = file_as_string.split('**PROTECTED[begin]')
                else:
                    unproc_patients = file_as_string.split('E_O_R')

                proc_patients = list()
                for p in range(len(unproc_patients)):
                    pat_i, record_id = process_xml_entry(unproc_patients[p],
                                                         list_of_tokens_to_remove,
                                                         list_of_token_to_replace)
                    if pat_i is not None:
                        proc_patients.append([pat_i, file[0:2] + '-' + record_id])
                    else:
                        nb_of_empty += 1

                data[file] = proc_patients
                print('{} patient(s)'.format(len(proc_patients)))
                print('{} empty patient(s)'.format(nb_of_empty))

    list_of_reports = []
    for k in data.keys():
        for pat, rec_id in data[k]:  # for all patients
            pat_str = []
            for t in TAGS:
                if t.startswith('TEXT'):
                    pat_str.append(pat[t])
            list_of_reports.append([" ".join(pat_str), rec_id])

    df_gt_epath = pd.read_csv(config.path_to_epath_file)

    # records = df_gt_epath['Record'].tolist()
    labeled_reports = list()
    cont = 0
    df_gt_epath['Record'] = df_gt_epath['Record'].str[:17]
    for report, rec_id in list_of_reports:
        print(rec_id)
        if rec_id in df_gt_epath['Record'].tolist():
            row = df_gt_epath[df_gt_epath['Record'] == rec_id]
            row = row.iloc[0]  # get only the first occurrence
            labeled_reports.append([rec_id, row['GTKum'],
                                    row['cell type'],
                                    row['ER'], row['PR'],
                                    report])
            cont += 1
    df_lab_rep = pd.DataFrame(labeled_reports,
                              columns=['REC_ID', 'GTKum',
                                       'Cell Type', 'ER',
                                       'PR', 'Report'])
    df_lab_rep['PR'] = df_lab_rep['PR'].replace('ND', '')
    df_lab_rep['ER'] = df_lab_rep['ER'].replace('ND', '')
    df_lab_rep['PR'] = df_lab_rep['PR'].replace('?', '')
    df_lab_rep['ER'] = df_lab_rep['ER'].replace('?', '')

    fname = os.path.join(config.path_to_preprocessed_reports,
                         'labeled_reports.csv')
    df_lab_rep.to_csv(fname, index=False)
    print('Cont: {}'.format(cont))
    # records = [rec[3:17] for rec in records]
    # print(df_gt_epath)

    # with open(config.path_to_preprocessed_reports, 'wb') as fh:
    # pickle.dump(df_lab_rep, fh)
