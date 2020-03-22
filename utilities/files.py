import os
import glob
import pydash
import random

def sample_files(num, path='data/LibriSpeech/test-clean'):
    random.seed(0)
    """Returns an array of random file names from a directory
    """
    audio_files = []
    num = int(num)
    files = list(glob.iglob(path + '/**/*.flac', recursive=True))
    audio_files = pydash.sample_size(files, num)
    print(audio_files[0:5])
    quit()
    
    return audio_files

    # while len(audio_files) < num:
    #     book_ids = os.listdir(path)
    #     book_ids = pydash.filter_(book_ids, lambda book_id: book_id.isnumeric())
    #     book_id = pydash.sample(book_ids)
    #     book_path = os.path.join(path, book_id)

    #     chapter_ids = os.listdir(book_path)
    #     chapter_ids = pydash.filter_(chapter_ids, lambda chapter_id: chapter_id.isnumeric())

    #     chapter_id = pydash.sample(chapter_ids)
    #     chapter_path = os.path.join(book_path, chapter_id)

    #     transcripts_list = [s for s in os.listdir(chapter_path) if s.endswith('.flac')]
    #     chosen_transcript = pydash.sample(transcripts_list)

    #     transcript_path = os.path.join(chapter_path, chosen_transcript)

    #     if not audio_files.__contains__(transcript_path):
    #         audio_files.append(transcript_path)

    # return audio_files


def get_transcript(audio_file):
    """Returns the transcript corresponding with an audio file"""
    parts = audio_file.split('/')
    parts[2] = 'test-clean'
    clean_path = '/'.join(parts)

    head_tail = os.path.split(clean_path)
    head = head_tail[0]
    tail = head_tail[1]
    file_name = tail.split('.')[0]


    book_id, chapter_id, transcript_id = file_name.split('-')
    trans_path = os.path.join(head, f'{book_id}-{chapter_id}.trans.txt')

    transcript_lines = open(trans_path, 'r').readlines()

    transcript_line = pydash.find(transcript_lines, lambda transcript_line: transcript_line.__contains__(f'{book_id}-{chapter_id}-{transcript_id}'))
    transcript_text = transcript_line.split(' ', 1)[1].strip()

    return transcript_text



def get_transcript_old(audio_file):
    """Returns the transcript corresponding with an audio file"""
    head_tail = os.path.split(audio_file)
    head = head_tail[0]
    tail = head_tail[1]
    file_name = tail.split('.')[0]

    book_id, chapter_id, transcript_id = file_name.split('-')
    trans_path = os.path.join(head, f'{book_id}-{chapter_id}.trans.txt')

    transcript_lines = open(trans_path, 'r').readlines()

    transcript_line = pydash.find(transcript_lines, lambda transcript_line: transcript_line.__contains__(f'{book_id}-{chapter_id}-{transcript_id}'))
    transcript_text = transcript_line.split(' ', 1)[1].strip()

    return transcript_text
