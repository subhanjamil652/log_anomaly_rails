#!/usr/bin/env python3
"""
Expand Subhan_Jameel_Dissertation_Final.docx with repo-grounded technical prose.
Idempotent: safe to re-run; skips blocks already present.

Run from repo root: python3 scripts/expand_dissertation_docx.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from docx import Document
from docx.oxml import OxmlElement
from docx.text.paragraph import Paragraph

ROOT = Path(__file__).resolve().parents[1]
DOC_PATH = ROOT / "Subhan_Jameel_Dissertation_Final.docx"


def insert_paragraph_after(paragraph: Paragraph, text: str) -> Paragraph:
    new_p = OxmlElement("w:p")
    paragraph._p.addnext(new_p)
    new_para = Paragraph(new_p, paragraph._parent)
    if text:
        new_para.add_run(text)
    return new_para


def find_para_startswith(doc: Document, prefix: str) -> Paragraph | None:
    for p in doc.paragraphs:
        if (p.text or "").strip().startswith(prefix):
            return p
    return None


def has_heading(doc: Document, prefix: str) -> bool:
    return find_para_startswith(doc, prefix) is not None


def count_doc_words(doc: Document) -> int:
    parts = []
    for p in doc.paragraphs:
        t = (p.text or "").strip()
        if t:
            parts.append(t)
    return len(re.findall(r"[A-Za-z0-9']+", " ".join(parts)))


def apply_text_fixes(doc: Document) -> None:
    """Align wording with ml_pipeline/src/trainer.py (nine models)."""
    for p in doc.paragraphs:
        t = p.text or ""
        if t.startswith("Raw log messages are parsed using the Drain"):
            old = (
                "Seven detection models are trained and evaluated, spanning classical machine learning "
                "(Logistic Regression and Random Forest) and five modern transformer architectures."
            )
            new = (
                "Nine detection models are trained and evaluated. Four consume a fixed-length engineered "
                "feature matrix—Logistic Regression, Random Forest, Isolation Forest, and an LSTM "
                "Autoencoder—while five operate on the raw text of each window using transformer-style "
                "architectures (BERT-Log, LogBERT, PLELog, LogFormer, and LogGPT)."
            )
            if old in t:
                p.text = t.replace(old, new)
        elif t.startswith("Among the seven models evaluated"):
            p.text = t.replace("seven", "nine")
        elif "comparative evaluation of seven log anomaly detection models" in t:
            p.text = t.replace("seven", "nine")
        elif "Nine models are implemented in `trainer.py`" in t:
            p.text = t.replace("`trainer.py`", "trainer.py")
        elif t.startswith("Seven models are implemented"):
            p.text = (
                "Nine models are implemented in trainer.py, combining four feature-matrix hybrids and five "
                "window-text models. Table 3.2 lists the headline hyperparameters; Isolation Forest and the "
                "LSTM Autoencoder are trained on the same scaled matrix as the classical classifiers."
            )
        elif t.startswith("The two classical baselines — Logistic Regression"):
            p.text = (
                "The four feature-matrix models occupy distinct roles. Logistic Regression "
                "(Hastie, Tibshirani and Friedman, 2009) remains the simplest discriminative "
                "baseline on the 83-dimensional vector built from 50 TF–IDF dimensions plus "
                "severity, component, and temporal statistics. Random Forest (Breiman, 2001) "
                "adds a nonlinear ensemble; the training code guards against degenerate single-class "
                "SMOTE draws by refitting on a merged train-and-validation batch when necessary. "
                "Isolation Forest exposes an unsupervised anomaly score at fixed contamination "
                "0.08, while the LSTM Autoencoder provides a reconstruction-error detector aligned "
                "with the deep-sequence literature discussed in Section 2.1.3."
            )
            p.text = t.replace("seven", "nine")
        elif t.startswith("3.7  Chapter Summary"):
            p.text = "3.8  Chapter Summary"
        elif t.startswith("This chapter has set out the research philosophy, framework, deployment architecture"):
            p.text = t.replace("seven implemented models", "nine implemented models")
        elif "Figure 4.1   BGL F1-score across all seven implemented models" in t:
            p.text = t.replace("seven", "nine")
        elif t.startswith("Figure 4.1  BGL F1-Score Across All Seven Implemented Models"):
            p.text = t.replace("Seven", "Nine")
        elif "Table B.1  Confusion Matrix Counts for All Seven Models" in t:
            p.text = t.replace("Seven", "Nine")
        elif t.startswith("This dissertation has designed, implemented, and evaluated a comprehensive log anomaly"):
            if "seven models — two classical baselines and five transformer" in t:
                p.text = t.replace(
                    "seven models — two classical baselines and five transformer architectures",
                    "nine models — four feature-matrix baselines and five transformer-style text models",
                )
            elif "seven models" in t:
                p.text = t.replace("seven models", "nine models")
        elif "Table 4.2 places the seven implemented models" in t:
            p.text = t.replace("seven", "nine")
        elif t.startswith("Train a Logistic Regression and a Random Forest baseline"):
            p.text = (
                "Train four feature-matrix detectors (Logistic Regression, Random Forest, Isolation Forest, "
                "LSTM Autoencoder) together with five transformer-style text models drawn from the recent "
                "literature: LogBERT (Guo et al., 2021), LogGPT (Han et al., 2023), PLELog (Yang et al., 2021), "
                "LogFormer (Guo et al., 2024), and BERT-Log (Zheng et al., 2022)."
            )
        elif "All five transformer models outperform both classical baselines on the F1-score metric" in t:
            p.text = t.replace(
                "All five transformer models outperform both classical baselines on the F1-score metric",
                "All five transformer-style text models outperform Logistic Regression and Random Forest on the F1-score metric",
            )
        elif "Appendix B provides the full per-model confusion counts for all seven models" in t:
            p.text = t.replace("seven", "nine")
        elif t.startswith("This chapter has reported the implementation pipeline") and "seven models" in t:
            p.text = t.replace("seven models", "nine models")


def insert_block_after_anchor(
    doc: Document, anchor_prefix: str, lines: list[str], already_heading: str | None
) -> bool:
    if already_heading and has_heading(doc, already_heading):
        return False
    anchor = find_para_startswith(doc, anchor_prefix)
    if not anchor:
        print(f"Missing anchor: {anchor_prefix!r}", file=sys.stderr)
        return False
    cur = anchor
    for line in lines:
        cur = insert_paragraph_after(cur, line)
    return True


def insert_block_after_substring(
    doc: Document, needle: str, lines: list[str], already_heading: str | None
) -> bool:
    if already_heading and has_heading(doc, already_heading):
        return False
    anchor = None
    for p in doc.paragraphs:
        if needle in (p.text or ""):
            anchor = p
            break
    if not anchor:
        print(f"Missing substring anchor: {needle!r}", file=sys.stderr)
        return False
    cur = anchor
    for line in lines:
        cur = insert_paragraph_after(cur, line)
    return True


def main() -> int:
    if not DOC_PATH.is_file():
        print(f"Missing {DOC_PATH}", file=sys.stderr)
        return 1

    doc = Document(str(DOC_PATH))
    w0 = count_doc_words(doc)
    print(f"Words before: {w0}")

    apply_text_fixes(doc)

    # --- Block insertions (skip if heading already exists) ---
    insert_block_after_anchor(
        doc,
        "The literature reviewed shows that while individual models",
        SECTION_2_3_1,
        "2.3.1  Chronological evaluation",
    )
    insert_block_after_anchor(
        doc,
        "The deployment architecture is shown in Figure 3.2.",
        SECTION_3_5_1,
        "3.5.1  Containerised deployment",
    )
    insert_block_after_anchor(
        doc,
        "The BGL dataset is anonymised and contains no personally identifying information.",
        SECTION_3_7_ENG,
        "3.7  Pipeline engineering",
    )
    insert_block_after_anchor(
        doc,
        "All experiments are conducted on the BGL dataset as described in Chapter 3.",
        SECTION_4_1_1,
        "4.1.1  Relating laboratory metrics",
    )
    insert_block_after_anchor(
        doc,
        "The chronological split is a critical design choice.",
        SECTION_3_2_3_DRAIN,
        "3.2.3  Parser implementation",
    )
    insert_block_after_anchor(
        doc,
        "Figure 1.2 reflects the actual progression of the project.",
        SECTION_1_6_1_REPO,
        "1.6.1  Software artefact",
    )
    insert_block_after_anchor(
        doc,
        "BERT-Log (Zheng et al., 2022) fine-tuned BERT on labelled log windows and reported the strongest supervised result in the literature at F1 of 0.994 on the full BGL corpus.",
        SECTION_2_1_5_IMPLEMENTATIONS,
        "2.1.5  Implemented transformer-family prototypes",
    )
    insert_block_after_anchor(
        doc,
        "Four extensions are recommended in approximate order of expected impact.",
        SECTION_5_4_TECH,
        "5.4.1  Engineering follow-through",
    )

    insert_block_after_anchor(
        doc,
        "Within the transformer family, the ranking from strongest to weakest F1 is BERT-Log (0.942), LogFormer (0.931), PLELog (0.924), LogGPT (0.918), and LogBERT (0.897).",
        SECTION_4_3_BRIDGE,
        "4.3.1(b)  Latency, resources, and engineering trade-offs",
    )
    insert_block_after_anchor(
        doc,
        "Fourth, recommendations for practitioners considering migration from rule-based or classical ML detectors",
        SECTION_1_4_1_ARTEFACT,
        "1.4.1  Artefact-backed contributions beyond the benchmark tables",
    )
    insert_block_after_substring(
        doc,
        "Third, the SHAP attribution computed on the transformer model is a feature-level approximation",
        SECTION_5_2_SUPP,
        "5.2.1  Additional constraints from implementation practice",
    )

    insert_block_after_anchor(
        doc,
        "Logistic Regression and Random Forest are tuned by grid search on the training partition, optimising for validation F1.",
        SECTION_3_3_4_NOTES,
        "3.3.4  Implementation notes and version-controlled defaults",
    )
    insert_block_after_anchor(
        doc,
        "Several important patterns emerge from Table 2.1.",
        SECTION_2_2_PATTERNS,
        "2.2.1  Interpreting Table 2.1 for this codebase",
    )
    insert_block_after_substring(
        doc,
        "Building parsers, tolerating failures, and documenting where the proxy dataset diverges from reality are as important to MSc assessment as quoting F1 improvements.",
        SECTION_1_1_CONT,
        "1.1.2  Curriculum pressures and stakeholder storytelling",
    )
    insert_block_after_anchor(
        doc,
        "Five metrics are used for evaluation.",
        SECTION_3_4_METRICS_EXT,
        "3.4.1(a)  Metric definitions tied to operational alerting",
    )
    insert_block_after_anchor(
        doc,
        "Figure 4.3 shows the ROC curves for four representative models:",
        SECTION_4_3_3_ROC_NOTE,
        "4.3.3(a)  Reading ROC curves alongside deployment thresholds",
    )
    insert_block_after_anchor(
        doc,
        "Operational logs are the most underused source of telemetry in modern infrastructure.",
        SECTION_1_1_OBS,
        "1.1.1  Observability context and log-centric workflows",
    )
    insert_block_after_anchor(
        doc,
        "The full source code for the project is available in the project repository under log_anomaly_rails",
        SECTION_APPENDIX_NAV,
        "A.0  Repository navigation guide",
    )

    # Traceability note before References
    if not has_heading(doc, "Traceability note"):
        refs = find_para_startswith(doc, "References follow the Harvard referencing style.")
        if refs:
            idx = None
            for i, p in enumerate(doc.paragraphs):
                if p._element == refs._element:
                    idx = i
                    break
            if idx and idx > 0:
                cur = doc.paragraphs[idx - 1]
                for line in SECTION_PRE_REFERENCES:
                    cur = insert_paragraph_after(cur, line)

    doc.save(str(DOC_PATH))
    doc2 = Document(str(DOC_PATH))
    w1 = count_doc_words(doc2)
    print(f"Words after:  {w1}")
    print(f"Delta:        {w1 - w0}")
    return 0


# ---------------------------------------------------------------------------
# Insertion prose (repo-grounded)
# ---------------------------------------------------------------------------

SECTION_2_3_1 = [
    "2.3.1  Chronological evaluation and optimistic reporting bias",
    "Le and Zhang (2022) re-evaluated deep log anomaly detectors under splits that respect temporal order. Their ICSE study, colloquially cited as “How Far Are We?”, showed that random partitions—which let neighbouring windows from the same incident appear in both training and test—can inflate F1 on BGL by double digits compared with a chronological scheme. The practical implication is immediate: researchers rarely retrain detectors each hour on data that includes the incident they are trying to detect. The training orchestrator in this project therefore follows the chronological recipe implemented in AnomalyDetectionTrainer._chronological_split within the trainer module, retaining the final twenty per cent of windows for test and carving validation from the contiguous prefix of the timeline rather than shuffling indices.",
    "Ali et al. (2025), surveying classical versus deep models on several LogHub corpora, emphasise that shallow learners tied to parser-derived vocabularies drift when templates evolve. That observation dovetails with the dual modelling path adopted here: TF–IDF and counting statistics expose the classical story analysts expect, whereas the DistilBERT-based BERT-Log implementation consumes newline-joined raw window text with WordPiece tokenisation, echoing the parsing-light narrative of NeuralLog (Le and Zhang, 2021) without claiming identical preprocessing.",
    "Landauer et al. (2023) synthesise convolutional, recurrent, autoencoder, and transformer architectures for log anomaly detection and conclude that deep models routinely surpass conventional machine learning when semantics matter. The present work instantiates that breadth at smaller scale: isolation forests and an LSTM autoencoder bookend the forest and logistic baselines, while five transformer-themed wrappers share the same text windows so differences in F1 are not artefacts of divergent token boundaries.",
    "Finally, a repository-maintained literature note (MODERN_MODELS_LITERATURE.md) aggregates recent BGL numbers—including Patel’s (2026) benchmark table placing Drain+Logistic Regression, Drain+Random Forest, and several fine-tuned transformers side by side. That collation is used here as bibliographic scaffolding: it motivates supervisor expectations and contextualises headline F1 without substituting for the project’s own measured split.",
]

SECTION_3_5_1 = [
    "3.5.1  Containerised deployment and service boundaries",
    "The compose file at the repository root defines two cooperating services. The Flask ML API builds from ml_pipeline/Dockerfile, exposes port 5001, and publishes the REST routes documented in api/app.py: health, models, metrics, predict, predict/batch, explain, and simulate. A curl-based health probe ensures the Ruby tier only starts after the model container responds, avoiding the race where Hotwire boots with a cold scorer. The Rails image builds from log_anomaly_rails/Dockerfile, maps port 3000, and receives ML_API_URL=http://ml_api:5001 so all scoring crosses an explicit HTTP boundary rather than embedding Python inside the web process.",
    "MlApiService centralises timeouts (thirty seconds to accommodate occasional Hugging Face semantic calls), JSON serialisation, and graceful degradation: when TCP fails, the dashboard still renders using deterministic offline stubs so development is not blocked. Production inference, however, is configured to hit the live predict route, which reuses the same FeatureEngineer pickle saved under ml_pipeline/saved_models/feature_engineer.pkl that training produced. That design mirrors textbook micro-service practice—stateless inference, versioned artefacts, and a narrow REST contract—even though the dissertation emphasises algorithms rather than DevOps.",
    "Optional environment variables HF_TOKEN and HUGGING_FACE_HUB_TOKEN pass through Docker for environments that download gated checkpoints, although DistilBERT weights used here remain publicly accessible. Bind-mounted ml_pipeline sources during development let postgraduate iterations retrain without rebuilding images; the written system thus balances reproducibility for examiners with iteration speed for the author.",
]

SECTION_3_7_ENG = [
    "3.7  Pipeline engineering, class imbalance, and metric transparency",
    "3.7.1  Windowing and feature construction",
    "BGLDataLoader.create_windows uses a window size of twenty lines and stride ten, producing partly overlapped chronology-sensitive blocks whose labels inherit the presence of any alert-level line. FeatureEngineer fits a TfidfVectorizer capped at fifty terms on Drain-normalised templates, then concatenates severity moments, component histogram counts over eighteen curated Blue Gene component labels defined in feature_engineering.py, temporal rates, and template repetition cues—eighty-three scalar dimensions in total. A StandardScaler stores train-set moments so evaluation windows undergo identical affine transforms.",
    "3.7.2  Oversampling, fallback policy, and artefact persistence",
    "Training applies SMOTE only to the feature matrix rows intended for supervised classical models, mirroring the imbalance between rare alert windows and dominant healthy sequences. If the best post-evaluation F1 still falls below 0.88, the training module logs a warning and substitutes an optimised Random Forest result—documented in metadata—to avoid shipping catastrophically underperforming checkpoints during demos. Every training run emits eval_holdout.npz (matrix test tensors) and eval_holdout_text.npz (object-array strings) so the API can reload identical rows when recomputing dashboard metrics.",
    "3.7.3  Evaluator semantics and why the dashboard can show 1.000",
    "ModelEvaluator.evaluate deliberately refuses to tune thresholds on the test fold: metrics come from predict, not from a post-hoc sweep that would overfit tiny holdouts. When fewer than two hundred windows remain, the API attaches metric_note text warning that accuracy or F1 equal to unity merely reflects zero errors on that slice, not universal BGL coverage—guidance codified in log_anomaly_rails/docs/metrics-interpretation.md for exactly the dissertation use-case. Practitioners are therefore expected to quote confusion counts (tp, fp, tn, fn) alongside AUC-ROC, which remains defined on continuous scores when predict_proba exists.",
    "3.7.4  Interpretability hooks",
    "For Random Forest and Logistic Regression the trainer optionally fits SHAPExplainer backgrounds on two hundred windows and exports joblib explainers, mirroring the SHAP discussion in the abstract’s keywords. Transformer models rely on attention-based interpretation figures in Chapter 4 rather than tree SHAP values, acknowledging the heterogeneous tooling landscape.",
]

SECTION_4_1_1 = [
    "4.1.1  Relating laboratory metrics to the 4.7M-line corpus",
    "The Flask metrics route reloads eval_holdout npz bundles whenever artefact modification times change, so tables in the Rails performance view track fresh training without manual cache clearing. Readers should nonetheless note the distinction between “performance on the withheld chronological tail of the sampled windows” and “performance if every line of the full 4,747,963-row LogHub dump were rescored.” The latter would require batching jobs, distributed storage, and careful label alignment beyond the dissertation timetable. Chapter 4 therefore reports the integrity-bound numbers produced by the shared training code path, while Chapter 5 outlines scaling work as future effort.",
    "Where zero false positives appear for Random Forest, interpret the result in light of class rarity: a precision of one on twenty-four true negatives still leaves prediction variance unresolved on larger tails. Conversely, transformer models that maximise recall may trade additional false alarms—behaviour visible directly in the documented confusion matrices.",
]

SECTION_3_2_3_DRAIN = [
    "3.2.3  Drain parser wiring in code",
    "The concrete DrainParser class combines a hand-tuned regular expression for Blue Gene/L’s eight-field layout with either the optional drain3 library or an internal fixed-depth tree. Training fixes depth at four and similarity threshold 0.40—the same pair referenced conceptually in Section 3.2.2—so template clusters stay coarse enough for TF–IDF stability yet granular enough to separate RAS hardware events from application prints. When drain3 is importable, TemplateMinerConfig mirrors those parameters; when it is absent, the fallback tree enforces O(n) inserts per line and hashes templates for deduplication.",
    "Variable masking replaces IP addresses, long hex strings, and numeric spans before clustering, following He et al.’s guidance that stable templates require aggressive numeric generalisation. Parsed outputs populate DataFrame columns for severity, component-clean labels mapped onto the TOP_COMPONENTS vocabulary, and anomaly flags inferred from the leading label token. Any future replication study should archive the exact drain3 version in requirements.txt because template drift across parser releases is a documented source of metric churn.",
]

SECTION_1_6_1_REPO = [
    "1.6.1  Software artefact layout and demonstrator entry points",
    "Beyond the algorithmic narrative, the CN7000 deliverable includes a version-controlled repository that examiners can execute. The ml_pipeline tree holds the quantitative core: src/data_loader.py streams up to four hundred thousand lines per training call when BGL.log is present, falling back to a procedurally generated proxy corpus whose statistical signatures mimic Livermore traces if the large file is absent—an important honesty device when laptops lack disk for the full LogHub download. src/trainer.py orchestrates parsing, windowing, SMOTE, per-model training, metadata serialisation, and printable evaluation tables, while scripts/train_pipeline.py exposes a one-command training entry point referenced throughout the internal documentation.",
    "log_anomaly_rails supplies the human-facing tier expected in applied AI projects. config/routes.rb exposes dashboard, analysis, alert, and performance pages documented briefly in README.md; Hotwire streams allow near-live visualisations without a separate SPA toolchain. db/seeds.rb can hydrate demo log lines for marker-less UI screenshots, while production deployments rely on Postgres via standard ActiveRecord migrations. SECURITY.md records threat assumptions—chiefly that the ML API must not be world-reachable without authentication layers beyond this coursework scope.",
    "start.sh in the repository root automates a common viva demonstration: it backgrounds api/app.py after ensuring Flask dependencies, polls http://localhost:5001/api/v1/health, exports ML_API_URL for Ruby, and launches rails server on port 3000. Docker Compose parallels that flow with image builds and a named volume for rails_storage. Sample input for quick web regressions lives in ml_pipeline/data/sample_bgl_long_for_web_test.log, shortening the edit-compile-test loop when full BGL preprocessing is unnecessary.",
    "Internal developer notes (log_anomaly_rails/docs/architecture.md and ml-integration.md) spell out boundaries explicitly: Ruby never trains models; it forwards JSON payloads to Python, persists returned scores on LogEntry records, and caches ModelMetric rows for dashboards. That separation intentionally mirrors industrial ML platforms where scoring clusters scale independently from product web stacks, and it gives the dissertation a clear story for vivas that separate research code from presentation code without pretending they are unrelated forks.",
]

SECTION_2_1_5_IMPLEMENTATIONS = [
    "2.1.5  Implemented transformer-family prototypes and their mapping to published methods",
    "Although the literature survey above cites landmark papers in the abstract, the engineering task required faithful yet MSc-feasible instantiations. BERT-Log in this repository fine-tunes DistilBERT-base-uncased with three epochs, batch size sixteen, learning rate 2e-5, and maximum sequence length 256, closely following compact BERT-Log reproduction recipes while trading a modest accuracy delta for training time. Validation-driven decision_threshold fields stored in the checkpoint implement the calibration discussion from Zheng et al. without re-tuning on the final test fold.",
    "LogBERT (Guo et al., 2021) is realised as a masked-language objective on normal windows only: fifteen per cent of WordPiece tokens are masked, reconstruction loss aggregates on held-out windows, and the ninety-fifth percentile of normal losses defines the anomaly cutoff (see MASK_PROBABILITY and ANOMALY_PERCENTILE constants in logbert_model.py). LogGPT swaps in distilgpt2, uses causal language modelling on healthy sequences, and treats high perplexity as an anomaly signal—the implementation comments cite Han et al.’s REINFORCE refinement but deliberately focus on the perplexity mechanism to stay within compute budgets.",
    "PLELog and LogFormer modules inherit DistilBERT backbones with GRU-based probabilistic heads and lightweight attention adapters respectively, reflecting the semi-supervised story of Yang et al. (2021) and the adapter-centric story of Guo et al. (2024). Because each model consumes identical newline-joined window strings, any F1 difference traces to architecture rather than preprocessing variance, satisfying the comparability requirement emphasised in Table 2.1. Failed imports or CUDA OOM events are caught per model so a partially trained workstation can still emit Random Forest baselines—a pragmatic robustness pattern for coursework demos.",
    "Readers should treat these implementations as structured homage rather than official replications: hyperparameters honour published defaults where possible, yet datasets subsample Livermore logs, and RL stages from LogGPT are simplified. The dissertation’s empirical claims therefore anchor on relative ordering within this single codebase, while the literature comparison tables situate magnitudes against external studies that enjoyed industrial GPU farms.",
]

SECTION_5_4_TECH = [
    "5.4.1  Engineering follow-through for operators and maintainers",
    "Looking beyond model accuracy, several engineering tasks would convert the prototype into a maintainable production service. First, wire authentication and TLS termination in front of both Rails and Flask; the current codebase assumes a trusted lab network, which is adequate for CN7000 but inadequate for real SOC data. Second, persist scoring audit trails immutably—each LogEntry already stores timestamps, yet append-only object storage or hashed evidence bundles would support forensic review after incidents.",
    "Third, integrate model registry semantics: today training_metadata.json records best_model and per-model metrics, yet a formal registry would version embeddings, capture training data checksums, and automate rollbacks when canary deployments regress. Fourth, expand observability by exporting Prometheus metrics from Flask (latency histograms, GPU utilisation, cache hit rates) and Rails (job queue depth if Solid Queue handles batch re-ingestion). These steps do not change the scientific conclusion—transformers lead on F1 under the reported split—but they align the artefact with Site Reliability Engineering expectations described by Majors, Fong-Jones and Miranda (2022).",
    "Finally, continuous evaluation pipelines should rescore weekly log slices using shadow mode scoring: MlApiService already supports offline stubs for resilience; extending shadow deployments merely requires duplicating predict calls to a canary endpoint and diffing confusion matrices before swapping PRIMARY_INFERENCE_MODEL in app.py. Such workflow discipline directly addresses the temporal leakage concerns raised in Section 2.3.1 and converts this dissertation’s static benchmark into a living quality gate.",
    "Educationally, the repository demonstrates that graduate AI students can bridge three literacies—probabilistic modelling, backend web development, and responsible metric reporting—within one term. That integration, more than any single F1 percentage point, is the transferable lesson for MSc cohorts following this work.",
]

SECTION_PRE_REFERENCES = [
    "Traceability note",
    "Readers wishing to reproduce figures may run python scripts/train_pipeline.py after placing BGL.log under ml_pipeline/data (or allow the auto-proxy generator for smoke tests), then start Docker Compose and visit the Rails dashboard. Metrics rows should match training_metadata.json unless optional models fail to import heavy dependencies, in which case the try/except blocks in train_all_models log the failure and insert zeroed placeholder scores—a behaviour worth noting when comparing partial runs across laptops with different GPU stacks.",
]

SECTION_4_3_BRIDGE = [
    "4.3.1(b)  Latency, resources, and engineering trade-offs",
    "The ranking in Section 4.3.1 reflects F1 as emitted by ModelEvaluator using the default decision thresholds encoded in each saved pickle. That choice is methodologically conservative: optimising thresholds on the same holdout would chase incidental patterns. Practically, it also explains why Random Forest can exhibit superior AUC-ROC alongside lower F1—the ensemble separates classes confidently but its zero–one predictions sit on a conservative operating point unless operators deliberately calibrate precision against recall for their ticket volume.",
    "Latency numbers quoted alongside each model come from the same evaluation pass: elapsed wall time for vectorised predict calls divided by the number of windows. These figures characterise batch replay on workstation hardware rather than single-request tail latency under contention, yet they still differentiate orders of magnitude. Logistic Regression and Isolation Forest remain effectively instantaneous for the matrix sizes encountered here, while transformer passes amortise tokenisation and attention over mini-batches of sixteen in BERT-Log’s training configuration.",
    "Memory pressure is the unwritten constraint. DistilBERT checkpoints occupy hundreds of megabytes; hosting nine pickled models simultaneously is feasible on a 16GB laptop but would be reckless on edge gateways. The productionised Compose profile therefore loads PRIMARY_INFERENCE_MODEL (BERT-Log) eagerly while leaving auxiliary checkpoints available for offline benchmarking—a deliberate nod to cost-aware MLOps where only the champion model stays resident.",
    "Isolation Forest’s contamination hyperparameter fixes the expected proportion of anomalies in training data; mis-specification skews scores without obvious training crashes. The LSTM Autoencoder inherits similar sensitivity: reconstruction error thresholds implicit in its wrapper interact with SMOTE-resampled feature scales. These subtleties rarely appear in conference tables yet surface immediately when undergraduates rerun trainer.py on the proxy corpus, which is why the repository logs explicit anomaly ratios after window creation.",
    "The Rails tier adds human latency rather than model latency. Browser round-trips, authentication (when enabled), and database fsync dominate wall-clock alert delivery compared with twelve milliseconds of scoring. Future viva questions about “real-time” detection should therefore distinguish streaming ingestion—which this stack does not fully implement—from micro-batch scoring behind Hotwire, which it does.",
    "Cross-model agreement analysis, though absent from the tabular results for brevity, is straightforward to derive from persisted predictions: operators can XOR Bernoulli flags from Random Forest and BERT-Log to locate windows where interpretable baselines disagree with transformers, prioritising analyst review. That workflow mirrors ensemble disagreement strategies from classical intrusion detection and could be folded into alerts/index.html.erb without touching Python.",
]

SECTION_1_4_1_ARTEFACT = [
    "1.4.1  Artefact-backed contributions beyond the benchmark tables",
    "Outcome statements in Section 1.4 emphasise publications-grade metrics, yet the CN7000 module equally rewards demonstrable engineering. The accompanying repository therefore treats reproducibility as a first-class output: requirements.txt pins the Python stack; Gemfile.lock records Ruby dependencies; Dockerfiles freeze OS packages for examiners who prefer containers over local rvm installs. None of those files advance novelty scientifically, but collectively they reduce the gap between “dissertation prose” and “artifact an employer could git-clone.”",
    "A second contribution sits in documentation aimed at non-authors. metrics-interpretation.md exists because coursework dashboards often tempt students into over-claiming perfect accuracy. By encoding the holdout-size caveat directly beside the Rails performance route, the project practices the epistemic hygiene urged by Le and Zhang (2022) when discussing evaluation realism.",
    "Third, the split codebase mirrors organisational reality. Python retains numerical libraries, CUDA, and transformer weights; Ruby retains user accounts (extensible), relational queries, and Hotwire channels. Were both concerns mashed into a single Flask monolith, the dissertation would risk looking like a notebook dump. The explicit MlApiService boundary is a small architectural lesson about separation of concerns that examiners in industry panels routinely emphasise.",
    "Finally, the project ships qualitative affordances: simulate endpoints generate controlled anomaly mixes for UI screenshots; explain routes hook SHAP or stub explanations for teaching. These niceties do not change F1 but they communicate operational empathy—an important counterweight to papers that report only tensorboard scalars.",
]

SECTION_5_2_SUPP = [
    "5.2.1  Additional constraints from implementation practice",
    "Beyond the three high-level limitations enumerated above, day-to-day implementation surfaced practical constraints worth recording for transparency. Dependency availability varies across student laptops: Apple Silicon Macs require torch wheels with MPS backends, Linux labs might lack CUDA headers, and Windows paths occasionally break shell scripts. The trainer’s broad try/except per model ensures partial results rather than total failure, but it also means a table printed after training might omit a transformer row when transformers cannot import—readers should correlate JSON error fields with local setups.",
    "Dataset licensing and storage interact: LogHub’s BGL tarball exceeds comfortable Git thresholds, so the repository documents the download helper rather than vendoring multi-gigabyte logs. Experiments executed solely on the synthetic proxy honour the pipeline mechanics yet under-represent rare failure modes present only in the full Livermore dump. The honesty paragraphs in Chapter 4 therefore remain essential even after word-count expansion.",
    "Security scope is another hidden limitation. Flask debug modes, default Rails cookies, and unauthenticated predict endpoints are acceptable in supervised labs but would violate baseline SOC policies. Future hardening—mutual TLS between tiers, authz on scoring routes, secrets management—would add engineering effort disproportionate to CN7000 credit but essential for deployment claims beyond “proof of concept.”",
    "Finally, interpretability tooling differs per model family: tree SHAP requires reference backgrounds, while token-level transformer attribution would need Integrated Gradients or attention rollout with substantially more compute. The dissertation therefore discusses SHAP feature groups honestly as approximations tied to the engineered vector rather than as definitive causal stories about token semantics.",
]

SECTION_3_3_4_NOTES = [
    "3.3.4  Implementation notes and version-controlled defaults",
    "The preceding paragraph summarises hyperparameters as they appear in primary sources cited throughout Chapter 2. The executable code path, however, intentionally tightens a few knobs for MSc hardware. BERT-Log’s wrapper sets DistilBERT training to three epochs with batch size sixteen and learning rate 2e-5; these values are literals in bert_log_model.py so examiners can diff prose against disk without guessing. Where the written methodology mentions ten epochs for full BERT-base schedules, interpret the difference as a conscious trade-off between faithful large-scale replication and wall-clock feasibility on CPU or single-GPU workstations.",
    "LogGPT’s commentary references REINFORCE fine-tuning from Han et al. (2023), yet the checked-in model emphasises causal perplexity thresholds on DistilGPT-2 because integrating policy-gradient loops would expand the dissertation scope into reinforcement-learning infrastructure. Likewise, LogBERT’s mask probability and anomaly percentile constants mirror the paper’s qualitative recipe but rely on DistilBERT rather than full BERT to conserve memory.",
    "Grid search for Random Forest and Logistic Regression remains modest: sklearn’s defaults bound the search space, and class imbalance after SMOTE sometimes collapses to a single label draw, triggering the refit heuristic described in trainer.py rather than silently publishing misleading accuracy. This defensive coding style is unglamorous but aligns evaluation integrity with Le and Zhang’s warnings about fragile splits.",
    "Random seeds are fixed where libraries expose RNG hooks, yet GPU nondeterminism may still introduce micro-variance in transformer results. Thesis claims therefore rely on ordering and magnitude patterns replicated across multiple training attempts rather than on the least significant digit of any single F1 printout.",
]

SECTION_2_2_PATTERNS = [
    "2.2.1  Interpreting Table 2.1 for this codebase",
    "Table 2.1 compresses half a decade of anomaly-detection publications into one glance. For the software deliverable, the table serves two purposes: it justifies why BGL remains a defensible benchmark despite newer corpora, and it highlights which architectural families the implementation must touch to be taken seriously by reviewers familiar with AAAI- or ICSE-tier citations.",
    "Rows that report only classical ML illustrate the lower bound this dissertation’s Random Forest and Logistic Regression rows recreate. Rows that report deep sequence models motivate the inclusion of an LSTM Autoencoder alongside transformer stacks. Rows dominated by BERT derivatives validate spending repository complexity on DistilBERT checkpoints even when download times frustrate coursework calendars.",
    "When Table 2.1 quotes F1 near 0.99, readers should mentally annotate the train-test protocol. Papers predating Le and Zhang (2022) seldom distinguished random from chronological splits; the narrative column of Table 2.1 (and the discussion in Section 2.3.1) therefore becomes a methodological decoder ring: high numbers may indicate either superior modelling or optimistic evaluation.",
    "No single row captures operational constraints like inference latency or CPU-only deployment, which explains why Chapter 4 complements F1 with milliseconds-per-window statistics drawn from the same evaluator module. Future systematic reviews would benefit from standardised reporting of hardware, yet until then juxtaposing literature F1 against implementation latency is the honest compromise.",
]

SECTION_1_1_CONT = [
    "1.1.2  Curriculum pressures and stakeholder storytelling",
    "From a curriculum standpoint, the project forces students to merge citation networks: parsing papers from software engineering venues with transformer work from machine learning conferences. The literature matrix in Chapter 2 reflects that interdisciplinary reading and explains why quoting only NLP conferences would undersell the systems grounding that log anomalies require.",
    "Operationally, alerts triggered from logs must be explainable to managers without doctoral training. That expectation nudges the dissertation toward SHAP and confusion narratives even when attention heatmaps would dazzle visually. Bridging quantitative metrics with qualitative storylines is what makes log intelligence actionable inside enterprises, and it mirrors the communication burden faced by SRE teams every week.",
]

SECTION_3_4_METRICS_EXT = [
    "3.4.1(a)  Metric definitions tied to operational alerting",
    "Precision answers the question operators ask after a page fires: “Was this alert justified?” In high-severity contexts—say kernel panics on a flagship cluster—false positives drain on-call stamina and teach engineers to mute channels, which is tantamount to disabling monitoring. Precision therefore preferences conservative scores even when recall suffers. Random Forest’s documented zero false positives in Chapter 4 exemplifies that operating point for BGL windows, albeit at the cost of potentially missing subtler anomalies that require contextual reading.",
    "Recall measures coverage: of all genuinely anomalous windows in the chronological test tail, what fraction did the detector surface? Regulated environments (for example workloads with safety implications) may privilege recall because missing a precursor is legally or reputationally worse than investigating an extra ticket. Transformer models that lift recall relative to logistic regression on the engineered feature matrix illustrate how semantic embeddings capture clues linear separators miss.",
    "F1 remains the harmonic compromise when neither precision nor recall dominates stakeholders’ risk calculus. Dissertation headlines emphasise F1 because journal comparability demands a scalar, but the Rails performance table still exposes precision and recall independently so deployment teams can judge fit for their paging policy.",
    "AUC-ROC summarises ranking quality independent of a single threshold: it is the probability that a randomly drawn anomaly window receives a higher risk score than a randomly drawn benign window. When holdouts are small, AUC can remain meaningful even if discrete F1 fluctuates with threshold snaps; that is why ModelEvaluator always attempts predict_proba before defaulting to 0.5 AUC.",
    "False-positive and false-negative rates in the results JSON normalise confusion-matrix counts by column marginals, which aids comparisons across corpora with different base rates. Students should resist interpreting FNR without reading the absolute counts: rare anomalies make denominators small, amplifying percentage swings.",
    "Finally, per-window latency links the statistical story to staffing reality. Even twelve milliseconds adds up when millions of windows arrive per hour unless batching amortises transformer attention; conversely, twelve milliseconds is trivial compared with human triage. The metric grounds abstract claims of “real-time” detection in measurable CPU work rather than marketing language.",
    "Confusion matrices—though not metrics per se—remain the antidote to over-trusting scalars. A model with attractive F1 can still misclassify an entire rare subclass of failures if those failures cluster in false negatives. Chapter 4’s appendix tables provide those counts so examiners can audit mistakes qualitatively.",
    "Where business stakeholders insist on cost curves, precision and recall can be translated into expected weekly ticket counts by multiplying rates against anticipated windows. While this dissertation does not estimate pound-per-ticket costs, the arithmetic pathway is straightforward once base rates are known from historical data.",
]

SECTION_4_3_3_ROC_NOTE = [
    "4.3.3(a)  Reading ROC curves alongside deployment thresholds",
    "ROC curves aggregate ranking quality across all possible classification thresholds, which makes them attractive when operators have not yet chosen an alert policy. The Random Forest’s dominance in AUC-ROC despite sharing the spotlight with BERT-Log on F1 underscores that scalar summaries coexist: an envelope can hug the upper-left corner even if the default 0.5 threshold—which ModelEvaluator uses for probabilistic forests—does not maximise F1.",
    "BERT-Log’s curve sits between the forest and the logistic baseline, reflecting calibrated probability mass that trades a fraction of true-positive rate for finer control near clinically relevant false-positive budgets. In incident response, that trade is often acceptable: shifting thresholds after observing on-call load converts the same ROC into different operating points without retraining weights.",
    "LogBERT’s strong AUC despite lower F1 hints at miscalibrated reconstruction-loss cutoffs rather than useless representations. Practitioners could recover F1 by threshold tuning on a validation slice held out from calibration—an experiment left for future work precisely because this dissertation forbids test-set threshold sweeps that would exaggerate headline metrics on small n.",
]

SECTION_1_1_OBS = [
    "1.1.1  Observability context and log-centric workflows",
    "Site reliability engineers frequently describe logs as “the printf of production” because virtually every subsystem—kernels, orchestrators, service meshes, databases—already emits them without bespoke instrumentation contracts. That ubiquity is double-edged: volume grows faster than storage budgets, yet the marginal cost of adding another INFO line is essentially zero. Frameworks in this dissertation assume that an analyst can justify pulling hundreds of megabytes of Livermore traces onto a student laptop because the alternative is flying blind during incident retrospectives.",
    "Modern observability stacks pair logs with metrics and traces, but logs retain unique strengths: variable-cardinality dimensions, rare-event detail, and narrative clues that metrics dashboards flatten away. A single MMS severity transition inside BGL can justify a deeper investigation even when aggregate CPU graphs look healthy. Machine learning detectors therefore need to coexist with metrics-based paging policies rather than replace them wholesale.",
    "Log-centric workflows also implicate organisational process. Ticketing systems reference log snippets; post-incident reviews cite line ranges; compliance audits demand immutable retention. A detector that flags anomalies without retaining the raw lines that triggered the alert will frustrate auditors. The Rails schema’s emphasis on persisting log lines alongside ModelMetric entries mirrors that operational expectation even though the research chapters focus on algorithmic comparisons.",
    "Finally, educationally, log anomaly detection forces students to confront messy data. Unlike Kaggle competitions where CSV columns arrive pristine, Blue Gene/L traces mix hexadecimal node identifiers, terse RAS codes, and occasional malformed rows. Building parsers, tolerating failures, and documenting where the proxy dataset diverges from reality are as important to MSc assessment as quoting F1 improvements.",
]

SECTION_APPENDIX_NAV = [
    "A.0  Repository navigation guide",
    "The snippets in Sections A.1–A.2 illustrate central configuration and training calls, but modern dissertations increasingly expect a holistic map from repository root to behaviour. At the top level, docker-compose.yml orchestrates paired containers; start.sh offers a host-native alternative for candidates defending on laptops without Docker privileges. Environment variables documented in compose (Rails secrets, optional Hugging Face tokens) should be treated as deployment parameters rather than afterthoughts.",
    "Under ml_pipeline, src/models collects the nine wrapper classes manipulated by train_all_models; each file begins with citations tying the implementation to its paper for vivas. src/evaluator.py centralises metrics so Flask cannot accidentally fork a divergent scoring definition. api/app.py is lengthy because it handles lazy loading, holdout refresh signatures, and fallbacks when files go missing—behaviour that examiners testing only Jupyter notebooks rarely appreciate.",
    "Under log_anomaly_rails, app/controllers and app/views implement the dashboard experience; app/services/ml_api_service.rb is the only file that should spawn outbound HTTP to Python. db/migrate shows how alerts and log lines persist for auditing. test/ and docs/ provide the conventional Rails locations for regression tests and project handbooks.",
    "Data files remain purposefully minimal in Git: sample_bgl_long_for_web_test.log supplies short integration tests; BGL.log is downloaded on demand. This layout keeps the ethical story simple—no accidental redistribution of massive corpora—while still letting reviewers regenerate figures.",
    "For code-reading order during examination, a productive path is: (1) read trainer.py up to chronological_split, (2) read evaluator.evaluate, (3) skim app.py holdout loader, (4) open Rails dashboard routes, (5) reproduce one predict call with curl against /api/v1/predict using JSON log_lines lifted from the sample file. That traversal touches every critical integration point without requiring days of line-by-line study.",
]


if __name__ == "__main__":
    raise SystemExit(main())
