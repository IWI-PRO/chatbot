import spacy

nlp = spacy.load("en_core_web_sm")

def chunk_text(text, chunk_size=100):
    """
    Split text into chunks of ~chunk_size words using spaCy sentence segmentation.
    """
    doc = nlp(text)
    chunks, current_chunk, current_len = [], [], 0

    for sent in doc.sents:
        words = sent.text.split()
        if current_len + len(words) <= chunk_size:
            current_chunk.append(sent.text)
            current_len += len(words)
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sent.text]
            current_len = len(words)

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks
