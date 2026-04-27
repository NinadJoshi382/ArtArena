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

Organize your dataset before running any pipeline scripts. For each artist–artwork pair, save all corresponding images in a dataset folder following this naming convention for each image:

```
<artwork>_<artist>
```

---

## Entry Trials

Entry Trials form the first stage of the ArtArena pipeline. The goal is to identify which artworks and artists are **deeply embedded** within the generative model — that is, cases where the model has internalized a style to a degree that raises concerns about potential memorization and style leakage. Artworks that pass this entry threshold are flagged as candidates for deeper evaluation in the Motif Duels stage.

> 💡 **Note:** How to use each script, which arguments to pass, and a usage example are documented in the comments at the very top of each file.

**Step 1 — Prepare Entry Trial inputs:**
```bash
python prep_ET.py \
        --dataset_dir   /path/to/dataset \
        --prompt_save   /path/to/prompts.pt \
        --mapping_json_save /path/to/mapping.json \
        --root_output_dir   /path/to/output \
        [--images_per_prompt 1] \
        [--name_split last]
```

**Step 2 — Run inference:**
```bash
  python ET_infer.py \
      --model_name sd15 \
      --prompts_pt /path/to/prompts.pt \
      --root_output_dir /path/to/output \
      --images_per_prompt 1 \
      --num_inference_steps 25 \
      --guidance_scale 7.5 \
      --seed_base 42
```

**Step 3 — Evaluate results:**
```bash
python ET_eval.py \
      --proximity_metric clip \
      --mapping_json /path/to/mapping.json \
      --output_csv /path/to/clip_scores.csv \
      --top_save_root /path/to/candidates \
      --orig_emb_out /path/to/original_clip_embs.pt \
      --gen_emb_out /path/to/generated_clip_embs.pt \
      --top_n 20
```

<!-- IMITATION FIGURE PLACEHOLDER -->
![Imitation Figure](figures/imitation.png)

---

## Motif Duels

Motif Duels form the second and more fine-grained stage of the pipeline. Here, the model is evaluated through structured **pariwise artwork interactions** in the following two roles:

- **Challenger** — artwork which contributes content cues via motifs in the compositon prompt.
- **Defender** — artwork which contributes stylsitic cues via explicit mention of the artwork and artist in the compositon prompt.

**Motif Extraction:** Motifs are extracted from reference artworks using **GPT-4o**, with the extraction prompt provided in the Appendix of the paper.

> 💡 **Note:** How to use each script, which arguments to pass, and a usage example are documented in the comments at the very top of each file.

**Step 1 — Prepare Motif Duel inputs:**
```bash
python prep_MD.py \
        --motif_json     /workspace/.../Motifs.json \
        --top20_dir      /workspace/.../top20_original \
        --MD_utils_dir   /workspace/.../MD_utils
```

**Step 2 — Run inference:**
```bash
  python MD_infer.py \
      --model_name sd15 \
      --out_json /path/to/clip_prompts_renamed.json \
      --suffix_pt /path/to/suffix.pt \
      --output_dir /path/to/output \
      --pair_dict_save_path /path/to/save_dict.pt \
      --images_per_prompt 1 \
      --num_inference_steps 25 \
      --guidance_scale 7.5
```

**Step 3 — Evaluate results:**
```bash
python MD_eval.py --metric clip \\
        --t1_path /path/to/Sets_prompt_dir_dict_learning.pt \\
        --artwork_map_path /path/to/simple_maps.pt \\
        --output_dir /path/to/output
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
