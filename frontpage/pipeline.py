from spacy.tokens import Span
from .utils import console


def dedup_stream(stream, key="text"):
    console.log("Deduplicating stream")
    uniq = {}
    for ex in stream:
        uniq[hash(ex[key])] = ex
    for ex in uniq.values():
        yield ex
    console.log("Deduplication finished")


def add_rownum(stream):
    for i, ex in enumerate(stream):
        yield {"text": ex["text"], "idx": i}


def attach_docs(lines, nlp, model):
    tuples = ((eg['text'], eg) for eg in lines)
    for doc, eg in nlp.pipe(tuples, as_tuples=True):
        eg['doc'] = sentence_classifier(doc, model)
        yield eg


def sentence_classifier(doc, model):
    doc.spans["sc"] = []
    for sent in doc.sents:
        preds = model(sent.text)
        for k, p in preds.items():
            if p >= 0.6:
                doc.spans["sc"].append(Span(doc, sent.start, sent.end, k))
                doc.cats[k] = max(doc.cats.get(k, 0.0), p)
    return doc


def attach_spans(stream, label, min_spans=1, max_spans=1):
    for ex in stream:
        spans = []
        for spansvals in ex['doc'].spans.values():
            for span in spansvals:
                spans.append(
                    {
                        "token_start": span.start,
                        "token_end": span.end - 1,
                        "start": span.start_char,
                        "end": span.end_char,
                        "text": span.text,
                        "label": label,
                    }
                )
        ex["spans"] = spans
        del ex["doc"]
        if len(spans) >= min_spans:
            if len(spans) >= max_spans:
                yield ex