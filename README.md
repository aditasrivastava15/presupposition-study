# Presupposition surprisal

This project tests whether open language models treat the presupposition of *stop* as background information. *Amy stopped smoking* assumes that Amy smoked before. *Amy did not stop smoking* still assumes that. *Mentioned* does not.

The scored word is *never*, in a second sentence that denies the habit:

- Amy stopped smoking. Amy had **never** smoked before.
- Amy did not stop smoking. Amy had **never** smoked before.
- Amy mentioned smoking. Amy had **never** smoked before.
- Amy did not mention smoking. Amy had **never** smoked before.

The comparison is surprisal of *never* after *stop* minus surprisal of *never* after *mentioned*. A positive difference means the denial is more surprising after *stop*.

## Result

On 24 held-out habits, before any training:

| Comparison | GPT-2 | Pythia-160M |
|---|---|---|
| *did not stop* minus *did not mention* | 0.47 bits, 24/24 habits | 0.56 bits, 23/24 habits |

Both confidence intervals exclude zero. That is the main result: the cost of denying the past habit survives negation.

An earlier test scored the verb after *used to* versus *has never*. Both *stopped* and *mentioned* became more surprising, so a negative context alone is not evidence of the presupposition. The affirmative *stopped* gap appears in GPT-2 and not in Pythia. The same denial test with *started* does not show the *stop* pattern.

After fine-tuning GPT-2 on 96 sentences with different names and habits, the *never* gap on the original 24 habits grows to about 9–11 bits. That is an effect of training, not a property of the original model.

## Files

| File | Contents |
|---|---|
| `sentences.csv` | 288 test sentences |
| `scores.csv` | GPT-2 before training |
| `scores_pythia-160m.csv` | Pythia-160M |
| `scores_gpt2-presupposition.csv` | GPT-2 after training |
| `surprisal_figure.png` | Bar chart of the GPT-2 conditions |
| `goal.txt` | Question, prediction, and result |
| `score_sentences.py` | Scores the critical word |
| `plot_results.py` | Draws the figure from `scores.csv` |
| `train_presupposition.py` | Fine-tunes GPT-2 on disjoint habits |

The trained weights are not in this repository. Run `train_presupposition.py` to recreate them.

## Run

Install PyTorch and Transformers, then from this directory:

```bash
python score_sentences.py
python score_sentences.py EleutherAI/pythia-160m
python plot_results.py
python train_presupposition.py
python score_sentences.py gpt2-presupposition
```

`score_sentences.py` writes `scores.csv` for GPT-2 and `scores_<model>.csv` otherwise.

## References

Parrish, A., Schuster, S., Warstadt, A., Agha, O., Lee, S. H., Zhao, Z., Bowman, S. R., & Linzen, T. (2021). NOPE: A corpus of naturally-occurring presuppositions in English. In *Proceedings of CoNLL 2021* (pp. 349–366).

Wilcox, E. G., Futrell, R., & Levy, R. (2024). Using computational models to test syntactic learnability. *Linguistic Inquiry, 55*(4), 805–848.
