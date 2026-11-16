"""Machine translation metrics: BLEU, chrF, TER (sacreBLEU) and METEOR (NLTK).

Usage:
    python translation_metrics.py         # runs the demo
    from translation_metrics import evaluate_translation
"""
import logging

import sacrebleu

logging.getLogger("sacrebleu").setLevel(logging.ERROR)


def evaluate_translation(hypotheses, references):
    """Corpus-level BLEU, chrF and TER.

    hypotheses : list of system translations
    references : list of reference translations (one per hypothesis)
    """
    bleu = sacrebleu.corpus_bleu(hypotheses, [references])
    return {
        "BLEU": bleu.score,
        "chrF": sacrebleu.corpus_chrf(hypotheses, [references]).score,
        "TER": sacrebleu.corpus_ter(hypotheses, [references]).score,
        "ngram_precisions": bleu.precisions,
        "brevity_penalty": bleu.bp,
    }


def sentence_scores(hypothesis, reference):
    """Sentence-level BLEU, chrF and METEOR (METEOR needs the NLTK WordNet data)."""
    scores = {"BLEU": sacrebleu.sentence_bleu(hypothesis, [reference]).score,
              "chrF": sacrebleu.sentence_chrf(hypothesis, [reference]).score}
    try:
        from nltk.translate.meteor_score import meteor_score
        scores["METEOR"] = 100 * meteor_score([reference.split()], hypothesis.split())
    except LookupError:
        scores["METEOR"] = None  # run: import nltk; nltk.download("wordnet")
    return scores


if __name__ == "__main__":
    refs = ["Le chat est assis sur le tapis.", "Il fait beau aujourd'hui."]
    systems = {
        "Good translation": ["Le chat est assis sur le tapis.", "Il fait beau aujourd'hui."],
        "Paraphrase": ["Le chat est installé sur le tapis.", "Le temps est agréable aujourd'hui."],
        "Bad": ["Chat tapis.", "Beau."],
    }
    for name, hyps in systems.items():
        s = evaluate_translation(hyps, refs)
        print(f"{name:17s} BLEU={s['BLEU']:5.1f}  chrF={s['chrF']:5.1f}  TER={s['TER']:5.1f}")

    print("\nParaphrase sensitivity (reference: 'the cat is sitting on the mat .')")
    ref = "the cat is sitting on the mat ."
    for hyp in ["the cat is sitting on the mat .", "the cat is seated on the mat .", "a cat sits on the rug ."]:
        s = sentence_scores(hyp, ref)
        print(f"  {hyp:34s}", {k: (round(v, 1) if v is not None else None) for k, v in s.items()})
