# Presentation Deck: Kolam Inverse — From Image to Equations

**Subtitle:** *Symbolic Image-to-Equation Reconstruction with Topological and Symmetry Invariance*  
**Presenter:** Neelanjana J. Anil  
**Domain:** Computer Vision, Ethnomathematics & Computational Geometry  

---

## Slide 01: Introduction & Cultural-Computational Context
* **Slide Title:** Introduction & Cultural-Computational Context
* **Slide Subtitle:** Bridging Traditional Ethnomathematics and Modern Geometric Inverse Graphics

### Slide Bullets
* **The Tradition of Kolam:** An ancient South Indian ritual drawing created with rice flour on doorsteps around rectilinear or hexagonal dot matrices (*pulli*).
* **The Computational Nature of Kolams:**
  * Not unconstrained sketches; they represent **continuous, non-intersecting Eulerian loops**.
  * They exhibit rigid planar graph embeddings, fractal scaling, and discrete dihedral group symmetries ($D_4$ and Klein-4 $V_4$).
* **The Inverse Problem:**
  * *Forward generation* is straightforward: apply a formal grammar or turtle-graphics sequence.
  * *Inverse reconstruction* is difficult: given only noisy image pixels, automatically infer the underlying symbolic equations (lines, arcs, Béziers) and grammatical rules.

### 🎤 Speaker Notes
> "Good morning. Today, I am presenting our work on Kolam Inverse. For centuries, Kolams have been practiced in South India as an ephemeral art form. To a computer scientist, however, a Kolam represents a rare convergence of discrete graph theory, continuous differential geometry, and formal language theory. Our objective is to solve the inverse problem: transforming raw pixel photographs into exact, continuous mathematical equations while strictly preserving topological continuity and group symmetry."

---

## Slide 02: Problem Statement & Motivation
* **Slide Title:** Problem Statement & Motivation
* **Slide Subtitle:** The Fundamental Failure of Autoregressive Neural Decoders in Geometric Inverse Graphics

### Slide Bullets
* **The Discrete-to-Continuous Semantic Gap:** Pixels are quantized on a sensor grid, while mathematical paths exist in continuous real coordinate space $\mathbb{R}^2$.
* **The Catastrophic Drift of Autoregressive Models:**
  * Sequence decoders (e.g., SketchRNN, autoregressive Transformers) predicting sequential coordinates $(x_t, y_t)$ suffer from **compounding positional drift** across long horizons ($700+$ steps).
  * Result: Cumulative errors cause the trajectory to miss closing vertices, creating open dangling loops and spurious jump strokes.
* **The Topological Blindness of Pixel Loss Functions:**
  * Standard losses (MSE, L1, Chamfer) only reward spatial closeness; they are mathematically blind to whether loops actually enclose their respective dot anchors.
  * Graph-theoretic invariants (even degree $d(v) \equiv 0 \pmod 2$, loop count $\beta_1$) are completely ignored.

### 🎤 Speaker Notes
> "Why does standard deep learning fail at this task? If you train a Transformer or an LSTM to output coordinate points sequentially, an error of just one pixel at step 30 snowballs by step 700. The loop fails to close, producing an invalid open curve. Furthermore, standard pixel-based loss functions are topologically blind: they do not penalize a model for breaking loop connectivity or missing dot anchors."

---

## Slide 03: Literature Review & Gap Analysis
* **Slide Title:** Literature Review & Gap Analysis
* **Slide Subtitle:** From Classical Formal Grammars to Deep Vector Graphics

### Comparison Table
| Domain | Prior Work & Approaches | Critical Gaps / Limitations |
| :--- | :--- | :--- |
| **Ethnomathematics & Formal Grammars** | Siromoney et al. (1974) array grammars; Marcia Ascher (2002) ethnomathematics; Robinson (2007) knot theory. | Strictly forward-generative paper models. No automated vision system to infer grammars from raw images. |
| **Autoregressive Sketch Synthesis** | SketchRNN (Ha & Eck, 2017), PolyGen (Nash et al., 2020), Co-Stroke (2022). | Severe compounding error over long steps ($>200$). Zero guarantee of closed Eulerian circuits. |
| **Differentiable Vector Graphics** | DiffVG (Li et al., 2020), LIVE (Ma et al., 2022), VectorFusion (2023). | Continuous rasterization optimization. Trapped in local minima; cannot discover discrete grammars or symmetry groups. |

### Our Technical Advantage
* **Topological guarantee:** Deterministic traversal on candidate lattice edges completely eliminates autoregressive drift.
* **Symbolic output:** Produces closed-form parametric and algebraic equations with bounded domains.
* **Group theory & grammar:** Recovers Klein-4 ($V_4$) dihedral symmetry and hierarchical Re-Pair grammar rules.

### 🎤 Speaker Notes
> "In the literature, research has been divided into two disconnected camps: formal linguists who wrote generative grammars on paper, and modern deep learning models that treat sketch curves as unconstrained bezier paths. Our work bridges this divide by using computer vision to extract candidate lattices, discrete mathematics to enforce Eulerian continuity, and continuous numerical optimization to fit parametric equations."

---

## Slide 04: Project Objectives & Scientific Ground Rules
* **Slide Title:** Project Objectives & Scientific Ground Rules
* **Slide Subtitle:** Rigorous Formulation, Blind Protocols, and Verifiable Milestones

### Core Objectives
1. **End-to-End Blind Extraction:** Build a pipeline processing raw input photos into continuous equations without human intervention.
2. **Dual-Model Empirical Benchmark:** Objectively compare a *Tuned Deterministic Lattice Baseline* against a *Deep Neural Edge Classifier (CNN+MLP)*.
3. **Closed-Form Mathematical Fitting:** Fit piecewise primitives (straight lines, circular arcs, cubic Béziers) using Orthogonal Distance Regression (ODR).
4. **Symmetry Orbit Reduction:** Prove and leverage Klein-4 ($V_4$) dihedral symmetry to achieve an exact $\approx 3\times$ parameter reduction.
5. **Hierarchical Grammar Induction:** Discover structural motifs via recursive digram pairing (Re-Pair).

### Scientific Ground Rules Enforced During Execution
* **Rule R1 (Script-Generated Numbers Only):** Zero hardcoded metrics. All figures come directly from logged execution.
* **Rule R2 (Blind Inference):** Test CSV files renamed to `csv_BLIND_LOCKED`; models had zero coordinate leakage.
* **Rule R3 (Clean Carve-Out):** 324 training, 36 validation, 40 held-out test designs.
* **Rule R4 (Negative & Identity Controls):** Synthetic black controls, shifted controls, and identity swap tests enforced.
* **Rule R5 (Transparent Reporting):** Negative results and model limitations reported honestly.

### 🎤 Speaker Notes
> "Our project adhered strictly to empirical research protocols. When testing our models, all ground-truth coordinate files were physically locked and inaccessible to the pipeline. Every metric was generated automatically by reproducible evaluation scripts, and all negative controls—such as testing against blank images or mismatched designs—were verified."

---

## Slide 05: System Workflow & Architecture
* **Slide Title:** System Workflow & Architecture
* **Slide Subtitle:** The Six-Stage Pipeline: From Pixels to Symbolic Geometry

### The Six Pipeline Stages
1. **Stage 01: Raw Input Capture** — Ingestion of calibrated pulli scans or blind uploads (RGB Image Matrix).
2. **Stage 02: Cleaned Mask** — Otsu thresholding + Area Opening ($\le 16\text{ px}$). *Verified: 0.0 px stroke skeleton erosion*.
3. **Stage 03: Medial Skeleton** — Zhang-Suen morphological thinning algorithm yielding a 1-pixel medial axis.
4. **Stage 04: Graph & Trajectory** — Junction detection ($N(p) \ge 3$) + Straight-through crossing rule ($\max \cos \theta$).
5. **Stage 05: Equation Fitting** — Orthogonal Distance Regression (ODR) for Lines, Arcs, and Cubic Béziers.
6. **Stage 06: Verification Overlay** — Euclidean distance transform + rasterization alignment (F1 / Chamfer Score & Overlay Map).

### 🎤 Speaker Notes
> "Here is our complete system architecture. Notice Stage 2: we use morphological opening to eliminate pulli dot anchors while rigorously confirming that exactly zero pixels of the stroke skeleton are eroded. In Stage 4, crossing junctions are traversed using the straight-through rule, maximizing cosine similarity between incoming and outgoing directions to prevent artificial kinks."

---

## Slide 06: Mathematical Foundations
* **Slide Title:** Mathematical Foundations
* **Slide Subtitle:** Continuous Differential Primitives, Dihedral Groups, and Grammar Induction

### 1. Closed-Form Primitive Formulations
* **Straight Line Segment:**
  * Parametric: $\mathbf{r}(t) = (1-t)\mathbf{p}_0 + t\mathbf{p}_1, \quad t \in [0, 1]$
  * Implicit Algebraic: $(y_1 - y_0)x - (x_1 - x_0)y + (x_1 y_0 - x_0 y_1) = 0, \quad x \in [x_{\min}, x_{\max}], y \in [y_{\min}, y_{\max}]$
* **Circular Arc:**
  * Parametric: $\mathbf{r}(t) = [h + r\cos(\theta_0 + t\Delta\theta), \; k + r\sin(\theta_0 + t\Delta\theta)]^T, \quad t \in [0, 1]$
  * Implicit Algebraic: $(x - h)^2 + (y - k)^2 = r^2$
* **Cubic Bézier Curve:**
  * Parametric: $\mathbf{B}(t) = (1-t)^3\mathbf{P}_0 + 3(1-t)^2 t\mathbf{P}_1 + 3(1-t)t^2\mathbf{P}_2 + t^3\mathbf{P}_3, \quad t \in [0, 1]$
  * Fitted via Levenberg-Marquardt non-linear least squares minimizing perpendicular residual distance.

### 2. Klein-4 ($V_4$) Dihedral Group Reduction
* Symmetry group $V_4 = \{ e, \sigma_x, \sigma_y, R_{180} \}$ acting on canvas coordinates with center $(c_x, c_y)$:
  $$\sigma_x(x, y) = (2c_x - x, y), \quad \sigma_y(x, y) = (x, 2c_y - y), \quad R_{180}(x, y) = (2c_x - x, 2c_y - y)$$
* **Exact Reduction:** Storing only the canonical seed quadrant $\mathcal{Q}_1$ allows regeneration of all 4 quadrants, reducing required storage by up to **2.94x**.

### 🎤 Speaker Notes
> "Every piece in our model is a rigorous mathematical object. Lines and arcs are provided with both parametric and algebraic implicit representations with bounded domains. For Béziers, we fit four control points via orthogonal distance regression. Using Klein-4 group theory, we mathematically prove that every dataset Kolam can be regenerated from its canonical seed quadrant, eliminating four-fold redundancy."

---

## Slide 07: Dataset & Lattice Geometry
* **Slide Title:** Dataset & Lattice Geometry
* **Slide Subtitle:** Three Canonical Pulli Families and Zero Edge Ambiguity

### Dataset Parameters
| Family | Grid Size | Total Candidate Edges | Dataset Designs | Mean Moves per Design | Minimum Edge Separation |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **kolam19** | $19 \times 19$ | 2,324 | 400 | 724 | **0.1118 lat units** |
| **kolam29** | $29 \times 29$ | 5,452 | 100 | 1,684 | **0.1118 lat units** |
| **kolam109** | $109 \times 109$ | 78,564 | 100 | 23,764 | **0.1118 lat units** |

### Empirical Findings on Lattice Geometry
* **Zero Candidate Edge Ambiguity (0.00%):** Across all 2,324 candidate lattice edges of kolam19, no edge midpoint lies within $0.1$ lattice units of any other edge. All candidate edges are disjoint and separable.
* **Symmetry Verification:** 100% of the true edge sets across all three families were verified to be strictly invariant under $V_4$ quadrant reflection.

### 🎤 Speaker Notes
> "We tested across three complete families from Kaggle's pulli dataset, scaling from a 19x19 grid up to a massive 109x109 lattice with over 23,000 moves. We proved mathematically that there is zero spatial ambiguity between candidate edges: the minimum separation between any two distinct candidate edges is 0.1118 lattice units, well above our 0.1 tolerance."

---

## Slide 08: Learning Framework & Experimental Audit
* **Slide Title:** Learning Framework & Experimental Audit
* **Slide Subtitle:** Deterministic Baseline vs. Deep Neural Edge Classifier

### 1. Model Architectures Tested Under Strict Blind Conditions
* **Tuned Deterministic Baseline:** Direct raster-skeleton correlation along candidate lattice paths. Hyperparameters tuned on validation: $\text{tol} = 0.10, \text{thresh} = 0.90$.
* **Neural Edge Classifier (CNN + MLP):** 3-layer Convolutional Feature Extractor + Spatial Coordinate Patch Sampler + 2-layer MLP classifier predicting edge existence $P(e \in E \mid I)$. Optimal threshold on validation: $\tau = 0.60$.

### 2. Comparative Results on 40 Blind Held-Out Designs
| Model | Edge Precision | Edge Recall | Edge $F_1$ | ROC-AUC | Move Match Rate | Correct Prefix | Chamfer |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Tuned Baseline** | **99.94%** | **99.94%** | **99.94%** | **1.0000** | **90.41%** | **654.5 / 724** | **0.131 lat** |
| **Neural Classifier** | 83.89% | 80.24% | **82.01%** | 0.9613 | 12.83% | 90.5 / 724 | 2.643 lat |

### 3. Key Research Insight
* The neural model achieved strong edge classification ($82.01\% F_1, 0.961 \text{ AUC}$), but minor edge dropouts break vertex degree parity, causing sequential traversal to terminate earlier ($90.5$ vs $654.5$ moves).
* **Conclusion:** In strictly constrained geometric lattice domains, deterministic projection vastly outperforms unconstrained neural models.

### 🎤 Speaker Notes
> "This slide presents a major finding of our research. The deep neural classifier reached an 82% F1 score and a 0.96 ROC-AUC under blind conditions. However, the deterministic baseline achieved 99.94% F1 and 90.4% move match. Why? In graph-constrained geometric lattices, exact coordinate correlation outperforms neural feature learning. Furthermore, even a 1% edge error in a neural model breaks Eulerian cycle continuity."

---

## Slide 09: Structural & Topological Validation
* **Slide Title:** Structural & Topological Validation
* **Slide Subtitle:** Image Verification, Chamfer Deviation, and the Generalization Test

### 1. Image-Level Verification (Rendered Equations vs Raw Skeletons)
| Family | Image $F_1$ @ 0.25 lat | Image $F_1$ @ 0.50 lat | Image $F_1$ @ 1.00 lat | Mean Chamfer Deviation | Negative Shift Control |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **kolam19** | 72.12% | **92.67%** | **99.69%** | **0.176 lat units** | 42.91% (Passed) |
| **kolam29** | 73.81% | **94.49%** | **99.64%** | **0.168 lat units** | 44.31% (Passed) |
| **kolam109** | 73.36% | **92.91%** | **99.83%** | **0.190 lat units** | 47.08% (Passed) |

### 2. Generalization on Arbitrary Uploads (The Complex Mandala Test)
* Tested on a non-Kolam, arbitrary 556-piece mandala drawing:
  * **Fit Fidelity ($F_1$ vs Extracted Skeleton):** **98.07%**
  * **Stroke Precision:** 98.39% | **Stroke Recall:** 97.75%
  * **Chamfer Distance:** **0.52 pixels** (sub-pixel alignment)

### 3. Scientific Transparency on Topology (Betti Numbers $\beta_0, \beta_1$)
* Raster Betti number comparison showed an anomaly ($\approx 43\%$ loop preservation) because the thresholded image mask contained pulli dots, while piecewise equations lacked post-fitting node snapping.
* **Honest Action:** Kept collapsed and explicitly marked *"under correction"* rather than presenting misleading claims.

### 🎤 Speaker Notes
> "When our fitted equations are rasterized and compared against the original image skeletons, we reach 92% to 94% F1 within half a lattice unit, and over 99.6% within one unit. Even on an arbitrary blind mandala upload with 556 pieces, our engine achieved 98.07% fit fidelity and a sub-pixel Chamfer distance of 0.52 pixels. Furthermore, when our Betti number comparison showed an anomaly due to open piecewise nodes, we honestly flagged it as under correction."

---

## Slide 10: Interactive Studio & System Implementation
* **Slide Title:** Interactive Studio & System Implementation
* **Slide Subtitle:** Dual-Interface Architecture and Real-Time Vector Synthesis

### 1. Dual Application Architecture
* **Engineering Baseline (`app.py` on `main`):** Lightweight, strictly focused on benchmark verification and tabular experimental logs.
* **Artsy Presentation Studio (`app_artsy.py` on `ui-artsy`):** A production-grade visual interface tailored for demonstration and cultural computing review.

### 2. Key Features of the Presentation Studio
* **Authentic Tamil Floor Palette:** Terracotta floor (`#0b0b14`), glowing rice-flour strokes (`#f6efe0`), saffron, turmeric, and Cormorant Garamond typography.
* **60 FPS Full-Vector Hero Animation:** Native SVG animation using `pathLength` and `stroke-dashoffset` across all 556 geometric primitives.
* **6-Stage Horizontal Journey:** Visual scroll-snap cards displaying the transformation from raw photo to verification overlay.
* **Interactive Comparison Slider:** Live Cyan (Input) / Magenta (Equations) / White (Overlap match) verification.
* **Live $V_4$ Symmetry Generator:** Visual animation of group reflections ($\sigma_x, \sigma_y, R_{180}$) fading in from the seed quadrant.
* **Equation Studio:** Dual parametric + implicit algebraic display with pagination and full JSON download.

### 🎤 Speaker Notes
> "To make our research accessible to both academic evaluators and the general public, we engineered two interfaces: a rigorous benchmark dashboard and an interactive presentation studio. The studio renders native vector SVGs at 60 FPS, provides an interactive before-and-after comparison slider, and allows users to export the complete set of mathematical equations in standard JSON format."

---

## Slide 11: Scope, Limitations & Critical Lessons
* **Slide Title:** Scope, Limitations & Critical Lessons
* **Slide Subtitle:** Honest Boundary Conditions and Failure Modes

### 1. The Superiority of Deterministic Algorithms in Structured Domains
* In unconstrained domains (e.g., natural images), deep neural networks dominate.
* In structured, graph-constrained geometric lattices, **deterministic algorithms with mathematical priors achieve 99.94% accuracy vs 82.01% for CNNs**.

### 2. Sensitivity of Planar Lattices to In-Plane Rotation
* The lattice edge classifier is robust to Gaussian noise ($\sigma=0.05, 79.56\% F_1$), blur ($1\text{px}, 80.31\% F_1$), and JPEG compression ($80.30\% F_1$).
* However, it is highly sensitive to camera rotation ($2^\circ \text{ tilt}$ drops $F_1$ to $31.36\%$) because rotated vertices drift away from orthogonal sampling points.

### 3. Real-World Mobile Photos Require Calibration
* The current pipeline is validated on calibrated, orthogonal scans. Real-world mobile photos with perspective tilt or non-uniform outdoor lighting require 4-point homography calibration before edge inference.

### 🎤 Speaker Notes
> "A critical lesson of our research is that deep learning is not always the best tool for every vision problem. When mathematical constraints are absolute, a hybrid architecture—using computer vision for feature extraction and discrete mathematics for traversal and fitting—guarantees topological validity. We also honestly acknowledge that real-world mobile photos with camera tilt will require perspective homography correction."

---

## Slide 12: Conclusion & Future Horizons
* **Slide Title:** Conclusion & Future Horizons
* **Slide Subtitle:** Summary of Contributions and Research Directions

### Summary of Contributions
* Demonstrated an end-to-end, blind inverse graphics pipeline translating discrete Kolam images into continuous closed-form equations.
* Achieved **99.94% edge detection accuracy** on 40 held-out designs and **92%–94% image F1** at half a lattice unit.
* Mathematically proved and verified an exact **2.89x to 2.94x parameter reduction** via Klein-4 ($V_4$) dihedral symmetry.
* Demonstrated up to **4.93x grammar compression** using Re-Pair hierarchical induction.
* Verified generalization on complex non-Kolam artwork (98.07% fit fidelity on a 556-primitive mandala).

### Future Research Directions
* **Mobile Perspective Rectification:** Automated 4-point homography estimation using pulli dots as vanishing point references.
* **Constrained Spline Optimization:** Global node-snapping to guarantee shared junction coordinates for 100% closed Betti loop counts.
* **Robotic Toolpath Synthesis:** Direct G-code compilation from induced equations for automated CNC sand plotters and physical drawing robots.

### 🎤 Speaker Notes
> "In conclusion, we have shown that traditional geometric art can be translated into exact, continuous mathematical equations with verifiable fidelity and symmetry invariance. This work preserves cultural heritage while advancing symbolic computer vision. Thank you, and I look forward to your questions."

---

## Slide 13 (Backup): Anticipated Faculty Q&A Defense Strategy
* **Slide Title:** Anticipated Faculty Q&A Defense Strategy
* **Slide Subtitle:** Prepared Strategic Responses for Reviewers

* **Q1: "Why did your CNN only achieve 82% when modern vision models easily get 95%+?"**  
  *Defense:* "Standard vision benchmarks classify high-level semantic categories where global texture suffices. Here, edge classification requires sub-pixel coordinate alignment. When edges are separated by only 0.11 lattice units, small receptive field misalignments cause false positives. The deterministic detector achieved 99.94% because it projects directly along the known lattice geometry."

* **Q2: "Why is the topology preservation box showing ~43% in the screenshots?"**  
  *Defense:* "The 43.7% Betti number preservation is an artifact of the raster comparison: the thresholded image mask contains pulli dots that create extra topological holes ($\beta_1$), while our piecewise equations are fitted per branch and have sub-pixel gaps at junctions before node snap-rounding. We recognized this discrepancy and honestly marked it 'under correction' rather than presenting misleading claims."

* **Q3: "Can your model handle a photo taken from an angle with a smartphone?"**  
  *Defense:* "Currently, the pipeline requires an orthogonal view. If taken at an angle, perspective foreshortening distorts regular lattice spacing. In future work, we plan to incorporate a 4-point homography estimation step using the pulli dots as vanishing point references to unwarp photos before feeding them to the equation engine."

* **Q4: "Why did you choose Klein-4 symmetry over standard rotation?"**  
  *Defense:* "While some Kolams have 8-fold dihedral symmetry $D_4$, all canonical pulli Kolams in this benchmark strictly obey the Klein-4 group $V_4 \cong \mathbb{Z}_2 \times \mathbb{Z}_2$ (horizontal and vertical reflections). We proved empirically that 100% of the true edge sets across all 600 designs are invariant under $V_4$, providing an exact 2.89x to 2.94x parameter reduction without loss."
