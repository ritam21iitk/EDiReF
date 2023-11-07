from config import *
import string
import nltk
import re

numbers = {
    "0":"zero",
    "1":"one",
    "2":"two",
    "3":"three",
    "4":"four",
    "5":"five",
    "6":"six",
    "7":"seven",
    "8":"eight",
    "9":"nine"
}

def remove_puntuations(txt):
    punct = set(string.punctuation)
    txt = " ".join(txt.split("."))
    txt = " ".join(txt.split("!"))
    txt = " ".join(txt.split("?"))
    txt = " ".join(txt.split(":"))
    txt = " ".join(txt.split(";"))
    
    txt = "".join(ch for ch in txt if ch not in punct)
    return txt

def number_to_words(txt):
    for k in numbers.keys():
        txt = txt.replace(k,numbers[k]+" ")
    return txt

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'_',' ',text)
    text = number_to_words(text)
    text = remove_puntuations(text)
    text = ''.join([i if ord(i) < 128 else '' for i in text])
    text = ' '.join(text.split())
    return text

def pad_to_len(list_data, max_len, pad_value):
    list_data = list_data[-max_len:]
    len_to_pad = max_len - len(list_data)
    pads = [pad_value] * len_to_pad
    list_data.extend(pads)
    return list_data

def get_emorynlp_vocabs(file_paths):
    emotion_vocab = vocab.Vocab()
    # keep 'neutral' in index-0
    emotion_vocab.word2index('neutral', train=True)
    for file_path in file_paths:
        data = json.load(open(file_path, 'r'), encoding='utf8')
        for episode in tqdm(data['episodes'],
                        desc='processing file {}'.format(file_path), disable=CONFIG['local_rank'] not in [-1, 0]):
            for scene in episode['scenes']:
                for utterance in scene['utterances']:
                    emotion = utterance['emotion'].lower()
                    emotion_vocab.word2index(emotion, train=True)
    if CONFIG['local_rank'] in [-1, 0]:
        torch.save(emotion_vocab.to_dict(), CONFIG['emotion_vocab'])
    logging.info('total {} emotions'.format(len(emotion_vocab)))

def get_meld_vocabs(file_paths):
    emotion_vocab = vocab.Vocab()
    # keep 'neutral' in index-0
    emotion_vocab.word2index('neutral', train=True)
    # global speaker_vocab, emotion_vocab
    for file_path in file_paths:
        data = pd.read_csv(file_path)
        for row in tqdm(data.iterrows(),
                        desc='get vocab from {}'.format(file_path), disable=CONFIG['local_rank'] not in [-1, 0]):
            meta = row[1]
            emotion = meta['Emotion'].lower()
            emotion_vocab.word2index(emotion, train=True)
    if CONFIG['local_rank'] in [-1, 0]:
        torch.save(emotion_vocab.to_dict(), CONFIG['emotion_vocab'])
    logging.info('total {} emotions'.format(len(emotion_vocab)))

def get_masac_vocabs(file_paths):
    emotion_vocab = vocab.Vocab()
    # keep 'neutral' in index-0
    emotion_vocab.word2index('neutral', train=True)
    # global speaker_vocab, emotion_vocab
    for file_path in file_paths:
        data = pd.read_csv(file_path)
        for row in tqdm(data.iterrows(),
                        desc='get vocab from {}'.format(file_path), disable=CONFIG['local_rank'] not in [-1, 0]):
            meta = row[1]
            emotion = meta['Emotion'].lower()
            emotion_vocab.word2index(emotion, train=True)
    if CONFIG['local_rank'] in [-1, 0]:
        torch.save(emotion_vocab.to_dict(), CONFIG['emotion_vocab'])
    logging.info('total {} emotions'.format(len(emotion_vocab)))


def build_dataset(dialogues, train=False):
    ret_utterances = []
    ret_labels = []
    for dialogue in dialogues:

        utterance_ids = []

        query = 'For utterance:'
        query_ids = tokenizer(query)['input_ids'][1:-1]

        for idx, turn_data in enumerate(dialogue):

            text_with_speaker = turn_data['speaker'] + ':' + turn_data['text']
            token_ids = tokenizer(text_with_speaker)['input_ids'][1:]
            utterance_ids.append(token_ids)


            if turn_data['label'] < 0:
                continue

            full_context = [CONFIG['CLS']]


            lidx = 0
            for lidx in range(idx):
                total_len = sum([len(item) for item in utterance_ids[lidx:]]) + 8
                if total_len + len(utterance_ids[idx]) <= CONFIG['max_len']:
                    break


            lidx = max(lidx, idx-8)
            for item in utterance_ids[lidx:]:
                full_context.extend(item)

            query_idx = idx
            prompt = dialogue[query_idx]['speaker'] + ' feels <mask>'
            full_query = query_ids + utterance_ids[query_idx] + tokenizer(prompt)['input_ids'][1:]
            input_ids = full_context + full_query
            input_ids = pad_to_len(input_ids, CONFIG['max_len'], CONFIG['pad_value'])
            ret_utterances.append(input_ids)
            ret_labels.append(dialogue[query_idx]['label'])

            if train and idx > 3 and torch.rand(1).item() < 0.2:
                query_idx = random.randint(lidx, idx-1)
                if dialogue[query_idx]['label'] < 0:
                    continue
                prompt = dialogue[query_idx]['speaker'] + ' feels <mask>'
                full_query = query_ids + utterance_ids[query_idx] + tokenizer(prompt)['input_ids'][1:]
                input_ids = full_context + full_query
                input_ids = pad_to_len(input_ids, CONFIG['max_len'], CONFIG['pad_value'])
                ret_utterances.append(input_ids)
                ret_labels.append(dialogue[query_idx]['label'])
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            print(dialogue[query_idx]['label'])
            print(turn_data['label'])
            print(input_ids)
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")

    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print(len(ret_utterances))
    
    print(len(ret_labels))
    print("________________________________________________________________")       
    dataset = TensorDataset(
        torch.LongTensor(ret_utterances),
        torch.LongTensor(ret_labels)
    )
    return dataset


def data2transformerencoder(dialogues, train=False):

    ret_utterances = []
    ret_labels = []
    max_len = 0
    for dialogue in dialogues:
        # utterance_ids = []

        
        # query = 'For utterance:'
        # query_ids = tokenizer(query)['input_ids'][1:-1]


        for idx, turn_data in enumerate(dialogue):

            ret_labels.append(dialogue[idx]['label'])

            pp_utt = preprocess_text(turn_data['text'])
            if pp_utt != "pad":
                marked_text = pp_utt
                tokenized_text = tokenizer.tokenize(marked_text)
                indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
                #tokens_tensor = torch.tensor([indexed_tokens])

            # text_with_speaker = turn_data['speaker'] + ':' + turn_data['text']
            # token_ids = tokenizer(text_with_speaker)['input_ids'][1:]
            # utterance_ids.append(tokens_tensor)
            # if turn_data['label'] < 0:
            #     continue
            # full_context = [CONFIG['CLS']]
            # lidx = 0
            # for lidx in range(idx):
            #     total_len = sum([len(item) for item in utterance_ids[lidx:]]) + 8
            #     if total_len + len(utterance_ids[idx]) <= CONFIG['max_len']:
            #         break
            # lidx = max(lidx, idx-8)
            # for item in utterance_ids[lidx:]:
            #     full_context.extend(item)

            # query_idx = idx
            # prompt = dialogue[query_idx]['speaker'] + ' feels <mask>'
            # full_query = query_ids + utterance_ids[query_idx] + tokenizer(prompt)['input_ids'][1:]
            # input_ids = full_context + full_query
            # input_ids = pad_to_len(input_ids, CONFIG['max_len'], CONFIG['pad_value'])

            CONFIG['max_len'] = 398
            CONFIG['pad_value'] = 0
            max_len = max(max_len,len(indexed_tokens))
            indexed_tokens = pad_to_len(indexed_tokens, 398, 0)
            ret_utterances.append(indexed_tokens)


            # print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            # print(dialogue[idx]['label'])
            # print(turn_data['label'])
            # print(indexed_tokens)
            # print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            
    # print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    # print(len(ret_utterances))
    # print(max_len)
    # print(len(ret_labels))
    # print("________________________________________________________________") 

    dataset = TensorDataset(
        torch.LongTensor(ret_utterances),
        torch.LongTensor(ret_labels)
    )
    return dataset

def get_iemocap_vocabs(file_paths):
    emotion_vocab = vocab.Vocab()
    emotion_vocab.word2index('neu', train=True)
    for file_path in file_paths:
        data = json.load(open(file_path, 'r'), encoding='utf8')
        for dialog in tqdm(data,
                desc='get vocab from {}'.format(file_path), disable=CONFIG['local_rank'] not in [-1, 0]):
            for utterance in dialog:
                emotion = utterance.get('label')
                if emotion is not None:
                    emotion_vocab.word2index(emotion, train=True)
    if CONFIG['local_rank'] in [-1, 0]:
        torch.save(emotion_vocab.to_dict(), CONFIG['emotion_vocab'])
    logging.info('total {} emotions'.format(len(emotion_vocab)))

def load_emorynlp_turn(file_path):
    emotion_vocab = vocab.Vocab.from_dict(torch.load(CONFIG['emotion_vocab']))
    data = json.load(open(file_path, 'r'), encoding='utf8')
    dialogues = []
    speaker_vocab = vocab.Vocab()
    for episode in tqdm(data['episodes'],
                    desc='processing file {}'.format(file_path), disable=CONFIG['local_rank'] not in [-1, 0]):
        for scene in episode['scenes']:
            dialogue = []
            for utterance in scene['utterances']:
                text = utterance['transcript']
                speaker = utterance['speakers'][0]
                speaker = speaker.split(' ')[0]
                emotion = utterance['emotion'].lower()
                emotion_idx = emotion_vocab.word2index(emotion)
                turn_data = {}
                turn_data['speaker'] = speaker
                speaker_vocab.word2index(speaker, train=True)
                turn_data['text'] = text
                turn_data['label'] = emotion_idx
                dialogue.append(turn_data)
            dialogues.append(dialogue)
    # speaker_vocab = speaker_vocab.prune_by_count(30)
    # for speaker_name in speaker_vocab.counts.keys():
    #     tokenizer.add_tokens(speaker_name)
    return dialogues


def load_meld_turn(file_path):
    emotion_vocab = vocab.Vocab.from_dict(torch.load(CONFIG['emotion_vocab']))
    data = pd.read_csv(file_path)
    pre_dial_id = -1
    dialogues = []
    dialogue = []
    speaker_vocab = vocab.Vocab()
    for row in tqdm(data.iterrows(),
                    desc='processing file {}'.format(file_path), disable=CONFIG['local_rank'] not in [-1, 0]):
        meta = row[1]
        text = meta['Utterance'].replace('’', '\'').replace("\"", '')
        speaker = meta['Speaker']
        emotion = meta['Emotion'].lower()
        emotion_idx = emotion_vocab.word2index(emotion)
        turn_data = {}
        turn_data['speaker'] = speaker
        speaker_vocab.word2index(speaker, train=True)
        turn_data['text'] = text
        turn_data['label'] = emotion_idx

        dialogue_id = meta['Dialogue_ID']
        if pre_dial_id == -1:
            pre_dial_id = dialogue_id
        if dialogue_id != pre_dial_id:
            dialogues.append(dialogue)
            dialogue = []
        pre_dial_id = dialogue_id
        dialogue.append(turn_data)
    dialogues.append(dialogue)
    # speaker_vocab = speaker_vocab.prune_by_count(30)
    # for speaker_name in speaker_vocab.counts.keys():
    #     tokenizer.add_tokens(speaker_name)
    return dialogues

def load_masac_turn(file_path):
    emotion_vocab = vocab.Vocab.from_dict(torch.load(CONFIG['emotion_vocab']))
    data = pd.read_csv(file_path)
    pre_dial_id = -1
    dialogues = []
    dialogue = []
    speaker_vocab = vocab.Vocab()
    for row in tqdm(data.iterrows(),
                    desc='processing file {}'.format(file_path), disable=CONFIG['local_rank'] not in [-1, 0]):
        meta = row[1]
        text = meta['Utterance'].replace('’', '\'').replace("\"", '')
        speaker = meta['Speaker']
        emotion = meta['Emotion'].lower()
        emotion_idx = emotion_vocab.word2index(emotion)
        turn_data = {}
        turn_data['speaker'] = str(speaker)
        speaker_vocab.word2index(speaker, train=True)
        turn_data['text'] = str(text)
        turn_data['label'] = emotion_idx

        dialogue_id = meta['Dialogue_ID']
        if pre_dial_id == -1:
            pre_dial_id = dialogue_id
        if dialogue_id != pre_dial_id:
            dialogues.append(dialogue)
            dialogue = []
        pre_dial_id = dialogue_id
        dialogue.append(turn_data)
    dialogues.append(dialogue)
    # speaker_vocab = speaker_vocab.prune_by_count(30)
    # for speaker_name in speaker_vocab.counts.keys():
    #     tokenizer.add_tokens(speaker_name)
    return dialogues


def load_iemocap_turn(file_path):
    emotion_vocab = vocab.Vocab.from_dict(torch.load(CONFIG['emotion_vocab']))
    data = json.load(open(file_path, 'r'), encoding='utf8')
    
    speaker_pools = json.load(open('./IEMOCAP/name_pool', 'r'))
    dialogues = []
    for dialog in tqdm(data,
            desc='processing file {}'.format(file_path), 
            disable=CONFIG['local_rank'] not in [-1, 0]):
        dialogue = []
        t_vocab = vocab.Vocab()
        speaker_vocab = vocab.Vocab()
        for utterance in dialog:
            speaker = utterance.get('speaker').upper()
            text = utterance.get('text').replace('[LAUGHTER]', '')
            emotion = utterance.get('label')
            speaker = speaker_pools[t_vocab.word2index(speaker, train=True)]
            speaker_vocab.word2index(speaker, train=True)
            turn_data = {}
            turn_data['speaker'] = speaker
            turn_data['text'] = text
            if emotion is not None:
                emotion_idx = emotion_vocab.word2index(emotion)
            else:
                emotion_idx = -1
            turn_data['label'] = emotion_idx
            dialogue.append(turn_data)
        dialogues.append(dialogue)
    # speaker_vocab = speaker_vocab.prune_by_count(30)
    # for speaker_name in speaker_vocab.counts.keys():
    #     tokenizer.add_tokens(speaker_name)
    return dialogues