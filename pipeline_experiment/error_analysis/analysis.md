# Error Analysis: LLM Pipeline for Conclusion Chapter Detection

## Overview

This error analysis examines LLM pipeline failures for detecting conclusion chapters in Electronic Theses and Dissertations (ETDs) based on ground truth data from 334 annotated pages across 299 ETDs. Analysis covers Stage 2 (chapter beginning page detection) and Stage 3 (conclusion classification) performance.

## Stage 2: Chapter Beginning Page Detection Coverage

The chapter beginning detection stage achieves coverage rates between 80.5% and 87.7% across all model configurations:

| Model Configuration | Coverage Rate |
|-------------------|---------------|
| llama33Results_stage2_3 | 87.7% (293/334) |
| llama33Results_full_pipeline | 87.4% (292/334) |
| llama4ScoutResults_stage2_3 | 86.5% (289/334) |
| llama4ScoutResults_full_pipeline | 86.2% (288/334) |
| llama3.18bResults_stage2_3 | 83.5% (279/334) |
| llama3.18bResults_full_pipeline | 81.4% (272/334) |
| mistralsmallResults_stage2_3 | 81.1% (271/334) |
| mistralsmallResults_full_pipeline | 80.5% (269/334) |

### Chapter Beginning Detection: Text Structure Patterns in Missed Pages

Analysis of 500-character content structure from `12_missed_chapter_pages_content_analysis.txt`:

1. **Embedded titles within continuous text flow**:
   - Page 82 (Bateman_TA_D_2017.pdf): Title appears mid-paragraph "69\ndata, implying that PM and vigilance may predict efficacy levels for the same tasks and that \nmeasurement items may, in fact, be contaminated with generalized self-efficacy. \nGeneral Discussion"
   - Page 91 (Bateman_TA_D_2017.pdf): Title embedded in flowing text "78\nsuch as creativity. Additional research will be required to clarify which motivational traits are the \nmost powerful predictors of self-regulation and performance in various situations. \nConclusions"

2. **Minimal structural hierarchy**:
   - Page 206 (Beskardes_GD_D_2017.pdf): Flat structure "Chapter 5\nConclusion\n185" with immediate page number, no subsections or explanatory text
   - Page 100 (Sreedharan_Nair_S_D_2014.pdf): Simple structure "87 \n \n8 \n \nConclusion \n \n8.1 Summary" with excessive whitespace

3. **Technical content dominating structure**:
   - Page 294 (NI_disseration.pdf): Immediate technical dive "Conclusions \n7.1. Introduction \nKnown for its influence to the performance of turbomachines, the tip leakage \nvortex is a dominating feature of the flow field near the rotor blade tip region"
   - Page 84 (WynnETD.pdf): Academic citation-heavy structure "76\nDISCUSSION\nTectonics\nEttensohn (1994) has suggested that the region was tectonically quiescent during deposition\nof the study interval"

4. **Non-standard formatting patterns**:
   - Page 100 (Waldron_CM_D_2017.pdf): Sparse layout "92 \n \nConclusion \n \n \nThe aerosolized experiments within the bio chamber and the walk-in refrigerator \nunit correlated well with one another"
   - Page 194 (Miller_DV_D_2017.pdf): Chapter number separation "184 \n \nConclusions \nDue to the prevalence of S-adenosyl-L-methionine (SAM) dependent enzymes"

## Stage 3: Conclusion Classification Performance

Stage 3 classification operates on the subset of pages correctly identified as chapter beginnings by Stage 2. Performance metrics show:

| Model Configuration | False Negatives | False Positives |
|-------------------|-----------------|-----------------|
| llama33Results_stage2_3 | 42 | 56 |
| llama33Results_full_pipeline | 40 | 53 |
| llama4ScoutResults_stage2_3 | 39 | 45 |
| llama4ScoutResults_full_pipeline | 37 | 44 |
| llama3.18bResults_stage2_3 | 33 | 41 |
| llama3.18bResults_full_pipeline | 27 | 35 |
| mistralsmallResults_stage2_3 | 19 | 30 |
| mistralsmallResults_full_pipeline | 15 | 24 |

### False Negative Analysis: Missed Conclusion Pages

From `missed_conclusions_content_analysis.txt`, 1 page is missed by all 8 model configurations:
- Page 58 (Seamans.pdf): "Chapter 3 - Data Collection, Analysis, Conclusions" with content "The focus of this study is the freshman college-student experience of needing information and the process of acquiring information"

From `correct_missed_conclusions_summary.csv`, 71 unique pages show false negatives across models. Content analysis reveals:

#### Error Rate Distribution from Content Analysis
- 100% error rate: 1 page (Seamans.pdf)
- 87.5% error rate: 2 pages (PFMDissertation.pdf page 325, MacLeanDissertation.pdf page 232)
- 75% error rate: 3 pages (Kook_Ph_D_Dissertation3.pdf page 191, JiangWang_dissertation_final.pdf page 151, JDSchiffbauerPhDGeosciences.pdf page 449)
- 62.5% error rate: 5 pages
- 50% error rate: 4 pages
- Lower error rates: Remaining 56 pages

#### Text Structure Analysis of False Negative Content Patterns

From `missed_conclusions_content_analysis.txt`, analysis of 500-character content structure:

1. **Hierarchical technical structure (high miss rates)**:
   - Page 325 (PFMDissertation.pdf, 87.5% miss rate): Structured as "CHAPTER TEN \n \n299\n10. CONCLUSIONS \n10.1 Single Airfoil in Turbulence \n10.1.1 Primary Contributions \n \nMeasurements of fluctuating surface pressure were made on a NACA 0015 airfoil"
   - Page 191 (Kook_Ph_D_Dissertation3.pdf, 75% miss rate): Format "Chapter 7 \n \n175\nChapter 7 \nConclusions and Implications \n7.1 Conclusions \nWhile the simulation based power systems analysis is one of the major parts in the field"

2. **Methodology-focused introductory structure**:
   - Page 58 (Seamans.pdf, 100% miss rate): Structure "Chapter 3 - Data Collection, Analysis, Conclusions \n \nFOCUS OF THE STUDY \nThe focus of this study is the freshman college-student experience of needing information \nand the process of acquiring information"
   - Page 166 (CaseStudyUpload4.pdf, 75% miss rate): Format "Chapter 5: Discussion of Findings, Conclusions, Limitations \nThe purpose of this study was to identify factors that make, an at-risk high school, \nso successful and to learn how it got there"

3. **List-based structural presentation**:
   - Page 125 (Dissertation_Prasun_Majumdar.pdf, 62.5% miss rate): Structure "111\nChapter 7: Conclusions and Recommendations \n7.1 Contribution to Bridge Deck Research \nThe following is a list of accomplishments from this dissertation research: \n7.1.1 Observations \n• The contact pressure distribution of real truck loading is non-uniform"
   - Page 103 (JS_thesis.pdf, 62.5% miss rate): Format "CHAPTER 8: CONCLUSIONS \n \nColumn Efficiency Measurements \nvan Deemter Plots \nvan Deemter curves were generated for the SpeedROD column using butylparaben"

4. **Sparse formatting with excessive whitespace**:
   - Page 194 (Fu_D_D_2010.pdf, 62.5% miss rate): Structure "Chapter 5 \nChapter 5  \nConclusion \n5.1 Conclusion \nThere is a large incentive to achieve high efficiency and high power density in power \ndelivery systems"
   - Page 232 (ETD_JuanjuanSun.pdf, 50% miss rate): Format "Chapter 5. Conclusions \n5.1 Conclusions \nTo meet the growing demand for higher power systems, as well as to supply the fast \nevolving powerful IC chips"

### False Positive Analysis: Incorrectly Classified Pages

From `fp_conclusions_by_all_models.csv`, 15 pages are misclassified as conclusions by all model configurations:

#### Universal False Positives (100% error rate across all models)
1. **Future Work chapters** (8 pages):
   - Page 305 (MSankirPhDDissertation.pdf): "Chapter8Suggested Future Research.pdf" - FUTURE_WORK label
   - Page 130 (dissertation_fong.pdf): "Chapter 8 Future Research" - FUTURE_WORK label
   - Page 147 (dissertation.pdf): "Chapter 6 Future Research" - FUTURE_WORK label
   - Page 298 (Layman.Dissertation_Finalc.pdf): "Chapter 13. Suggested Future Work" - FUTURE_WORK label
   - Page 259 (Disser99.pdf): "CHAPTER 9. RECOMMENDATIONS FOR FUTURE WORK" - FUTURE_WORK label
   - Page 95 (Han_Y_D_2016.pdf): "CHAPTER 6: FUTURE WORK" - FUTURE_WORK label
   - Page 112 (Bowden_ZE_D_2016.pdf): "Chapter 5: Summary and Directions for Future Research" - SUMMARY|FUTURE_WORK label
   - Page 137 (Munger_ZW_D_2016.pdf): "CHAPTER 5. DISSERTATION SUMMARY AND FUTURE RESEARCH" - FUTURE_WORK|SUMMARY label

2. **Recommendations chapters** (3 pages):
   - Page 161 (Amara_dissertation.pdf): "Chapter 6. RECOMMENDATIONS" - RECOMMENDATIONS label
   - Page 118 (MokaremDissertation.pdf): "CHAPTER 8: RECOMMENDATIONS" - RECOMMENDATIONS label
   - Page 152 (Taheriandani_M_D_2016.pdf): "Chapter 10 Summary and Recommendations" - SUMMARY|RECOMMENDATIONS label

3. **Summary/Discussion chapters** (4 pages):
   - Page 136 (DISSERTATION_FINAL_JDW.pdf): "Chapter 8: Summary & Directions for Future Research" - SUMMARY|FUTURE_WORK label
   - Page 140 (SullivanDissertation.pdf): "CHAPTER 5: SUMMARY AND DISCUSSION" - DISCUSSION|SUMMARY label
   - Page 131 (Onefilegood.pdf): "Chapter VII: SUMMARY" - SUMMARY label

#### Text Structure Analysis of False Positive Content Patterns

From `false_positive_conclusions_content_analysis.txt`, analysis of structural elements that trigger misclassification:

1. **Research outcome discussion structure** (Future work misclassified as conclusions):
   - Page 305 (MSankirPhDDissertation.pdf): Structure "286\n \n \nCHAPTER 8 \n \nSuggested Future Research \n \nThe disulfonated polysulfone copolymers always resulted in with lower \nfuel/oxidant permeability compare to Nafion™. However, the large water dependence \nof the proton conductivity of copolymers makes these membranes limited at higher \ntemperatures"
   - Page 130 (dissertation_fong.pdf): Format "Chapter 8 Future Research \nN.H. Ben Fong \nChapter 8 Future Research \n \n \nIn studying classical control theory, a mathematical model of a dynamic system is defined as a \nset of equations that adequately predicts the behavior of a system"

2. **Directive/prescriptive structure** (Recommendations misclassified as conclusions):
   - Page 161 (Amara_dissertation.pdf): Structure "Chapter 6. Recommendations \n147 \nAmara Loulizi \nChapter 6. RECOMMENDATIONS \n  \nBased on the findings of the present study, the following recommendations are made: \n• \nThe effects of overlays on the air-coupled and ground-coupled GPR system \nwaveforms need to be studied"
   - Page 118 (MokaremDissertation.pdf): Format "107\nCHAPTER 8: RECOMMENDATIONS \n \n1. \nThe unrestrained shrinkage test method, ASTM C 157, may be used as a performance \nbased specification for restrained concrete systems"

3. **Goal-oriented introductory structure** (Summary misclassified as conclusions):
   - Page 131 (Onefilegood.pdf): Structure "Chapter VII: SUMMARY \nThe purpose of this series of studies had two goals. First, to identify \nmeans of reducing the time required for subterranean termites to discover \ncommercial bait stations. Second, to identify strategies for reducing the \nlikelihood that subterranean termites would desert a commercial bait station"
   - Page 140 (SullivanDissertation.pdf): Format "CHAPTER 5: SUMMARY AND DISCUSSION \n \nWith the No Child Left Behind Act's (NCLB's) increased emphasis on the subjects of math, \nscience, and English/reading, some educators voiced concerns about the de-emphasis of \ninstruction in other areas like art, music, physical education"

4. **Forward-looking structural cues** (All false positive categories):
   - Future work pages start with project-oriented language: "The disulfonated polysulfone copolymers always resulted...", "In studying classical control theory..."
   - Recommendations use directive language: "Based on the findings...recommendations are made", "may be used as a performance based specification"
   - Summaries use retrospective goal language: "The purpose of this series of studies had two goals", "increased emphasis on the subjects of math"

## Individual Model Text Structure Analysis

### Model Performance Ranking by Structure Sensitivity

From `5_conclusion_detection_metrics_on_detected_chapters.csv`, models ranked by structural robustness (stage2_3 configuration):

1. **mistralsmallResults_stage2_3**: 19 FN, 30 FP - best structural detection
2. **llama3.18bResults_stage2_3**: 33 FN, 41 FP - moderate structural sensitivity  
3. **llama4ScoutResults_stage2_3**: 39 FN, 45 FP - higher structural confusion
4. **llama33Results_stage2_3**: 42 FN, 56 FP - highest structural sensitivity

### Model-Specific Text Structure Handling Patterns

From `missed_conclusions_content_analysis.txt`, individual model responses to structural patterns:

#### Mistral Small: Most Structure-Robust Model
**Unique successful detections** (where other models fail):
- Page 232 (MacLeanDissertation.pdf): Successfully handles "Chapter 7, Section 1: Overview of Significant Results \npage - 211 \n \n \n7.1 Overview of Significant Results \n \n \nThe results presented here have been the outcome of a broad effort" while 7 other models miss this complex hierarchical structure
- Page 449 (JDSchiffbauerPhDGeosciences.pdf): Only mistralsmall_full_pipeline successfully detects "429\nCHAPTER 6 \n \nClosing thoughts \n \nJAMES D. SCHIFFBAUER \nDepartment of Geosciences" minimal content structure

**Structure pattern strengths**: 
- Handles sparse formatting better (multiple successful detections on whitespace-heavy pages)
- Less sensitive to embedded author information and institutional affiliations
- More robust to non-standard chapter numbering patterns

#### LLaMA 3.3: Most Structure-Sensitive Model  
**Consistent structural failures** (misses across both pipeline variants):
- Page 325 (PFMDissertation.pdf): Fails on hierarchical technical structure "CHAPTER TEN \n \n299\n10. CONCLUSIONS \n10.1 Single Airfoil in Turbulence" (missed by both llama33 variants)
- Page 232 (MacLeanDissertation.pdf): Cannot handle complex section referencing "Chapter 7, Section 1: Overview of Significant Results \npage - 211"
- Page 191 (Kook_Ph_D_Dissertation3.pdf): Struggles with multi-page chapter headers "Chapter 7 \n \n175\nChapter 7 \nConclusions and Implications"

**Structure pattern weaknesses**:
- Sensitive to technical subsection hierarchies (10.1, 10.1.1 patterns)
- Confused by page number references within chapter headers
- Poor handling of duplicated chapter title formatting

#### LLaMA 4 Scout: Moderate Structural Adaptation
**Mixed performance patterns**:
- Successfully handles some sparse structures that confuse LLaMA 3.3
- Page 100 (Sreedharan_Nair_S_D_2014.pdf): llama4Scout_stage2_3 successfully detects "87 \n \n8 \n \nConclusion \n \n8.1 Summary" while llama4Scout_full_pipeline fails
- Inconsistent performance across pipeline variants on same structural patterns

#### LLaMA 3.1 8B: Technical Structure Specialist
**Specific structural advantages**:
- Page 194 (Miller_DV_D_2017.pdf): Only llama33Results_stage2_3 successfully handles "184 \n \nConclusions \nDue to the prevalence of S-adenosyl-L-methionine (SAM) dependent enzymes" technical biochemical content
- Better performance on domain-specific technical terminology within structural context

### Pipeline Variant Structure Handling Differences

#### Full Pipeline Structural Improvements
- **llama3.18b**: 6-point improvement in both FN and FP (33→27 FN, 41→35 FP) suggests full pipeline better handles complex multi-label structures
- **mistralsmall**: 4-point FN improvement (19→15) indicates full pipeline processing enhances sparse structure detection
- **llama33 and llama4Scout**: Minimal improvements (2-point FN reduction) suggest these models have structural processing limitations that pipeline stages cannot overcome

#### Stage2_3 vs Full Pipeline Structural Sensitivity
From model comparison, full_pipeline consistently shows:
- Better handling of multi-label chapter titles ("Conclusions and Implications", "Discussion and Conclusion")
- Improved detection of sparse formatting patterns  
- Enhanced processing of embedded structural elements within 500-character limit

### Model-Specific False Positive Structure Patterns

From `false_positive_conclusions_content_analysis.txt`, individual model structural confusion patterns:

#### Universal Structural Confusion (All Models)
**Directive structure misclassification** - All 8 model configurations misclassify pages with:
- Numbered recommendation structures: "1. The unrestrained shrinkage test method, ASTM C 157, may be used as a performance based specification" (Page 118, MokaremDissertation.pdf)
- Bullet-point directive formatting: "• The effects of overlays on the air-coupled and ground-coupled GPR system waveforms need to be studied" (Page 161, Amara_dissertation.pdf)  
- Goal-statement introductions: "The purpose of this series of studies had two goals" (Page 131, Onefilegood.pdf)

#### Model-Specific Structural Advantages in False Positive Reduction
**LLaMA 3.1 8B**: Shows some resistance to certain false positive structures:
- Page 112 (Bowden_ZE_D_2016.pdf): llama3.18bResults_full_pipeline correctly avoids misclassifying "Chapter 5: Summary and Directions for Future Research" while all other models fail
- Page 100 (Sreedharan_Nair_S_D_2014.pdf): llama3.18bResults_full_pipeline correctly handles "Table of contents page" classification while stage2_3 variant fails

**Mistral Small**: Most consistent structural discrimination:
- Lower false positive rates (24-30 vs 44-56 for other models) suggest better structural boundary detection
- More accurate handling of multi-label structures that confuse other models

#### Structural Elements Triggering Universal Misclassification
**Research outcome discussion patterns**: All models misclassify content starting with:
- "The disulfonated polysulfone copolymers always resulted in with lower fuel/oxidant permeability"
- "In studying classical control theory, a mathematical model of a dynamic system is defined"
- "Based on the findings of the present study, the following recommendations are made"

**Retrospective goal structures**: All models confused by:
- "The purpose of this series of studies had two goals"
- "With the No Child Left Behind Act's (NCLB's) increased emphasis on the subjects of math, science, and English/reading"

## Summary of Text Structure Findings

1. **Stage 2 structural detection failures**: 269-293 out of 334 pages detected with 4 structural patterns causing misses - embedded titles within text flow, minimal hierarchical structure, technical content dominance, and non-standard formatting with excessive whitespace

2. **Stage 3 false negative structure patterns**: 15-42 pages missed per model showing hierarchical technical structures (87.5% miss rates), methodology-focused introductions (100% miss for Seamans.pdf), list-based presentations, and sparse formatting patterns

3. **Stage 3 false positive structure patterns**: 15 pages universally misclassified showing research outcome discussion structures (future work), directive/prescriptive structures (recommendations), goal-oriented introductory structures (summaries), and forward-looking structural cues across all categories

4. **Model-specific structural capabilities**: Mistral Small shows best structural robustness (19-30 errors vs 42-56 for LLaMA 3.3), successfully handling sparse formatting and embedded author information. LLaMA 3.3 shows highest structural sensitivity, failing on technical hierarchies and duplicated formatting. LLaMA 3.1 8B demonstrates some structural discrimination advantages in full_pipeline variant.

5. **Pipeline variant structural improvements**: Full_pipeline consistently improves structural handling across models, with 1-6 point error reductions, particularly effective for multi-label chapter titles and sparse formatting patterns within 500-character processing constraint