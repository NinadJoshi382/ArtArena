# ArtArena

Official repository for **SilentBrush**, a framework for evaluating the stylistic influence of artistic styles in text-to-image generative models.

---

## Abstract

> *SilentBrush* investigates whether large-scale text-to-image models silently memorize and reproduce the distinctive visual styles of living and recently deceased artists without attribution or consent. We introduce a two-stage evaluation pipeline — **Entry Trials** and **Motif Duels** — that systematically probes models for style leakage at both the holistic and fine-grained levels. Entry Trials identify artworks and artists whose styles are deeply embedded within a model's weights, serving as candidates for further scrutiny. Motif Duels then stage a structured visual competition between a challenger model (prompted to imitate a target artist) and a defender model (prompted generically), isolating the presence of artist-specific visual motifs that persist beyond coincidence. Together, these two stages provide a rigorous and interpretable benchmark for quantifying unauthorized style memorization in generative models.

---

## Teaser

<!-- TEASER FIGURE PLACEHOLDER -->
![Teaser Figure](figures/teaser.png)

---

## Dataset Preparation

Organize your dataset before running any pipeline scripts. For each artist–artwork pair, save all corresponding images in a dedicated folder following this naming convention:

```
<artwork>_<artist>/
```

For example:
```
starry_night_vangogh/
water_lilies_monet/
the_scream_munch/
```

All images belonging to the same artwork–artist pair should reside within their respective folder. Ensure folder names are lowercase and use underscores in place of spaces.

---

## Entry Trials

Entry Trials form the first stage of the ArtArena pipeline. The goal is to identify which artworks and artists are **deeply embedded** within the generative model — that is, cases where the model has internalized a style to a degree that raises concerns about potential memorization and style leakage. Artworks that pass this entry threshold are flagged as candidates for deeper evaluation in the Motif Duels stage.

> 💡 **Note:** How to use each script, which arguments to pass, and a usage example are documented in the comments at the very top of each file.

**Step 1 — Prepare Entry Trial inputs:**
```bash
python prep_ET.py
```

**Step 2 — Run inference:**
```bash
python ET_infer.py
```

**Step 3 — Evaluate results:**
```bash
python ET_eval.py
```

<!-- IMITATION FIGURE PLACEHOLDER -->
![Imitation Figure](figures/imitation.png)

---

## Motif Duels

Motif Duels form the second and more fine-grained stage of the pipeline. Here, the model is evaluated through structured **head-to-head visual competitions** between two roles:

- **Challenger** — the model is prompted to explicitly generate an image *in the style of* a target artist.
- **Defender** — the same model is prompted with a generic, style-agnostic description of the same subject.

The duel asks: does the Challenger's output contain artist-specific visual motifs that the Defender's does not? If identifiable motifs persist in the Challenger's generations beyond what can be attributed to the subject matter alone, this constitutes evidence of style memorization.

**Motif Extraction:** Motifs are extracted from reference artworks using **GPT-4o**, with the extraction prompt provided in full in the Appendix of the paper.

> 💡 **Note:** How to use each script, which arguments to pass, and a usage example are documented in the comments at the very top of each file.

**Step 1 — Prepare Motif Duel inputs:**
```bash
python prep_MD.py
```

**Step 2 — Run inference:**
```bash
python MD_infer.py
```

**Step 3 — Evaluate results:**
```bash
python MD_eval.py
```

<!-- DUEL FIGURE PLACEHOLDER -->
![Duel Figure](figures/duel.png)

---

## Citation

If you find this work useful, please consider citing:

```bibtex
@article{silentbrush2025,
  title     = {SilentBrush: Probing Artistic Style Memorization in Text-to-Image Models},
  author    = {},
  journal   = {},
  year      = {2025}
}
```

---

## License

This project is licensed under the [MIT License](LICENSE).
