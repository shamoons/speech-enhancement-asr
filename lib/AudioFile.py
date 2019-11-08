import os
import pydash
from soundfile import SoundFile


class AudioFile:
    def __init__(self, data_path='data/LibriSpeech'):
        books_path = os.path.join(data_path, 'BOOKS.txt')
        books_content = open(books_path, 'r').readlines()
        self.DATA_PATH = data_path
        self.BOOKS = []
        for book in books_content:
            book_split = pydash.map_(book.split('|'), lambda x: x.strip())
            self.BOOKS.append(book_split)

    def load_sound_file(self, audio_type, book_id, chapter_id, transcript_id):
        sound_file = SoundFile(os.path.join(self.DATA_PATH, audio_type, book_id, chapter_id,
                                            f'{book_id}-{chapter_id}-{transcript_id}.flac'))
        return sound_file

    def load(self, book_id, chapter_id, transcript_id, subset='test-clean'):
        def check_transcript_id(t_line):
            t_id = t_line.split(' ', 2)[0].split('-')[2]
            if t_id == transcript_id:
                return True
            return False

        transcript_lines = open(os.path.join(self.DATA_PATH, 'test-clean', book_id,
                                             chapter_id, f'{book_id}-{chapter_id}.trans.txt'), 'r').readlines()
        transcript_line = pydash.find(transcript_lines, check_transcript_id)
        transcript_text = transcript_line.split(' ', 1)[1].strip()

        return {
            'subset': subset,
            'transcript_text': transcript_text,
            'book_id': book_id,
            'chapter_id': chapter_id,
            'transcript_id': transcript_id,
            'clean_sound_file': self.load_sound_file(subset, book_id, chapter_id, transcript_id),
            # 'dev-noise-gaussian-5': self.load_sound_file('dev-noise-gaussian-5', book_id, chapter_id, transcript_id)
        }

    def load_random(self, subset='test-clean'):
        book_ids = os.listdir(os.path.join(self.DATA_PATH, subset))
        book_id = pydash.sample(book_ids)
        clean_path = os.path.join(self.DATA_PATH, subset, book_id)

        chapter_id = pydash.sample(os.listdir(clean_path))
        chapter_path = os.path.join(clean_path, chapter_id)
        transcripts_list = [s for s in os.listdir(chapter_path) if s.endswith('.flac')]
        chosen_transcript = pydash.sample(transcripts_list)

        transcript_id = chosen_transcript.split('.')[0].split('-')[2]

        return self.load(book_id, chapter_id, transcript_id, subset=subset)
