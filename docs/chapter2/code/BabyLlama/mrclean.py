import re

# START_TOKEN = '<s>'
# END_TOKEN = '</s>'
# PADDING_TOKEN = '<pad>'

START_TOKEN = ''
END_TOKEN = ''
PADDING_TOKEN = ''

def _make_padding_sequence(seq_length):
    return ''.join([END_TOKEN] + seq_length * [PADDING_TOKEN])

def cleanup_simple_wikipedia(text, seq_length):
    pad_seq = _make_padding_sequence(seq_length)
    text = START_TOKEN + re.sub(r'\n\n', pad_seq + START_TOKEN, text) + pad_seq
    return text

def cleanup_wikipedia(text, seq_length):
    pad_seq = _make_padding_sequence(seq_length)
    text = re.sub(r'= = = (.+?) = = =\n', r'\1', text)
    lines = [line.strip() for line in text.splitlines()]
    text = START_TOKEN + re.sub(r'\n\n', pad_seq + START_TOKEN, '\n'.join(lines)[1:]) + pad_seq
    return text

def cleanup_qed(text, seq_length):
    # TODO: this should probably be padded too, but it’s difficult to detect when subtitles start and end
    # The handling of proper nouns and of parentheses isn’t perfect, but this is still an improvement over the base text
    punctuation_ex = re.compile(r'([.!?]\s*)')
    unimportant_chars_ex = re.compile(r'\(.*?\)|[.!?]')
    lines = []
    for line in text.splitlines():
        nchars = len(line)
        if nchars > 0:
            line_body = unimportant_chars_ex.sub('', line)
            f_upper = sum(c.isupper() for c in line_body) / len(line_body)
            if f_upper >= 0.5: # Mostly uppercase characters
                # Taken from https://stackoverflow.com/a/41662260
                split_on_punctuation = punctuation_ex.split(line.replace('l', 'I'))
                line = ''.join([sentence.capitalize() for sentence in split_on_punctuation])
        lines.append(line.strip())
    return START_TOKEN + '\n'.join(lines) + END_TOKEN + ''.join(seq_length * [PADDING_TOKEN])

def cleanup_extra_spaces(text):
    multiple_spaces_ex = re.compile(r'[ \t\u00A0]+')
    space_before_punctuation_ex = re.compile(r'[ \t\u00A0]([.,;!?])')
    text = multiple_spaces_ex.sub(' ', text)
    text = space_before_punctuation_ex.sub(r'\1', text)
    return text

def cleanup_bnc_spoken(text, seq_length):
    pad_seq = _make_padding_sequence(seq_length)
    text = cleanup_extra_spaces(text)
    text = START_TOKEN + re.sub(r'\n\n', pad_seq + START_TOKEN, text) + pad_seq
    return text

def cleanup_aochildes(text, seq_length):
    text = cleanup_extra_spaces(text)
    return START_TOKEN + text + _make_padding_sequence(seq_length)

def cleanup_cbt(text, seq_length):
    text = cleanup_extra_spaces(text)
    space_before_apostroph = re.compile(r"([\w\d])[ \t\u00A0](['’]\w)")
    #space_before_quote = re.compile(r"[ \t\u00A0](['’])")
    #space_after_quote = re.compile(r"([`])[ \t\u00A0]")
    #text = space_before_quote.sub(r'\1', text)
    #text = space_after_quote.sub(r'\1', text)
    text = space_before_apostroph.sub(r'\1\2', text)
    return START_TOKEN + text + _make_padding_sequence(seq_length)

def cleanup_children_stories(text, seq_length):
    # Sometimes one skipped line marks the beginning of a new story,
    # but sometimes it is present within a same story, which doesn’t
    # make it very useful for separating independent stories.
    return START_TOKEN + text + _make_padding_sequence(seq_length)

def cleanup_gutenberg(text, seq_length):
    # Overall, the text is clean, however some entries don’t seem
    # very useful, e.g. figure captions preceded by a number.
    # Not sure if we should remove them, because that would also
    # remove bullet lists which are otherwise consistent with the
    # surrounding text.
    # No start or end tokens because the text seems to be cut.
    return text + ''.join(seq_length * [PADDING_TOKEN])

def cleanup_open_subtitles(text, seq_length):
    # The text is mostly clean, apart from some subtitle credits
    # such as "Subtitles by ...".
    subtitle_credit_ex = re.compile(r'^.*subtitle.*$\n', re.MULTILINE | re.IGNORECASE)
    text = subtitle_credit_ex.sub('', text)
    return START_TOKEN + text + _make_padding_sequence(seq_length)

def cleanup_switchboard(text, seq_length):
    # No start or end tokens because the text seems to be cut.
    return text + ''.join(seq_length * [PADDING_TOKEN])