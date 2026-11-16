"""Sequence labeling metrics (NER, BIO scheme): token accuracy vs entity-level precision / recall / F1.

An entity counts as correct only if its boundaries AND its type are exact (CoNLL standard).
The functions below implement it from scratch; seqeval is used to cross-check when installed.

Usage:
    python sequence_labeling_metrics.py   # runs the demo
    from sequence_labeling_metrics import entity_scores, token_accuracy
"""


def get_entities(tags):
    """Convert a BIO tag sequence into a set of entities (type, start, end)."""
    entities, start, etype = [], None, None
    for i, tag in enumerate(list(tags) + ["O"]):
        boundary = tag == "O" or tag.startswith("B-") or (tag.startswith("I-") and tag[2:] != etype)
        if boundary:
            if etype is not None:
                entities.append((etype, start, i - 1))
            start, etype = (i, tag[2:]) if tag != "O" else (None, None)
    return set(entities)


def token_accuracy(y_true, y_pred):
    """Share of tokens with the correct tag (inflated by the many easy 'O' tags)."""
    pairs = [(t, p) for st, sp in zip(y_true, y_pred) for t, p in zip(st, sp)]
    return sum(t == p for t, p in pairs) / len(pairs)


def entity_scores(y_true, y_pred):
    """Entity-level precision, recall and F1 over a list of sentences."""
    tp = fp = fn = 0
    for t, p in zip(y_true, y_pred):
        gold, pred = get_entities(t), get_entities(p)
        tp += len(gold & pred)
        fp += len(pred - gold)
        fn += len(gold - pred)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def error_types(y_true, y_pred):
    """Count boundary errors, type errors, missed and spurious entities."""
    counts = {"boundary error": 0, "type error": 0, "missed": 0, "spurious": 0}
    overlap = lambda a, b: not (a[2] < b[1] or a[1] > b[2])
    for t, p in zip(y_true, y_pred):
        gold, pred = get_entities(t), get_entities(p)
        for g in gold - pred:
            ov = [q for q in pred - gold if overlap(g, q)]
            if not ov:
                counts["missed"] += 1
            elif any(q[0] == g[0] for q in ov):
                counts["boundary error"] += 1
            else:
                counts["type error"] += 1
        counts["spurious"] += sum(1 for q in pred - gold if not any(overlap(q, g) for g in gold))
    return counts


if __name__ == "__main__":
    sentence = ["Emmanuel", "Macron", "visite", "la", "Banque", "de", "France", "à", "Paris"]
    y_true = [["B-PER", "I-PER", "O", "O", "B-ORG", "I-ORG", "I-ORG", "O", "B-LOC"]]
    y_pred = [["B-PER", "I-PER", "O", "O", "B-ORG", "O", "O", "O", "B-LOC"]]  # ORG truncated

    print("Sentence        :", " ".join(sentence))
    print("Gold entities   :", sorted(get_entities(y_true[0])))
    print("Pred entities   :", sorted(get_entities(y_pred[0])))
    print("Token accuracy  :", round(token_accuracy(y_true, y_pred), 3))
    print("Entity scores   :", {k: round(v, 3) for k, v in entity_scores(y_true, y_pred).items()})
    print("Error types     :", error_types(y_true, y_pred))
    try:
        from seqeval.metrics import f1_score
        print("seqeval F1      :", round(f1_score(y_true, y_pred), 3))
    except ImportError:
        print("(install seqeval to cross-check: pip install seqeval)")
