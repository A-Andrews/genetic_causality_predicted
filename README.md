## Columns in Data

| Column | Description |
|--------|-------------|
| `Ancient_Sequence_Age_Human_Enhancer` | Age of human enhancer sequence |
| `Ancient_Sequence_Age_Human_Promoter` | Age of human promoter sequence |
| `BLUEPRINT_DNA_methylation_MaxCPP` | Max causal probability for DNA methylation QTLs |
| `BLUEPRINT_H3K27acQTL_MaxCPP` | Max causal probability for H3K27ac QTLs |
| `BLUEPRINT_H3K4me1QTL_MaxCPP` | Max causal probability for H3K4me1 QTLs |
| `BP` | The position of the variant on the chromosome (duplicate of pos) |
| `Backgrd_Selection_Stat` | Background selection statistic |
| `BivFlnk` | Bivalent flanking regions |
| `CHR` | Which chromosome the variant is on (duplicate of chrom) |
| `CM` | Genetic distance used in linkage studies |
| `CTCF_Hoffman` | CTCF binding site from Hoffman et al. |
| `Coding_UCSC` | Is the variant in a coding region |
| `Conserved_LindbladToh` | Conserved region from Lindblad-Toh |
| `Conserved_Mammal_phastCons46way` | Conservation across mammals |
| `Conserved_Primate_phastCons46way` | Conservation across primates |
| `Conserved_Vertebrate_phastCons46way` | Conservation across vertebrates |
| `CpG_Content_50kb` | CpG content in 50kb window |
| `DGF_ENCODE` | DNase I genomic footprint from ENCODE |
| `DHS_Trynka` | DNase hypersensitive site from Trynka |
| `Enhancer_Hoffman` | Enhancer regions from Hoffman et al. |
| `FetalDHS_Trynka` | Fetal DNase hypersensitive site |
| `GERP.NS` | GERP conservation score (neutral substitutions) |
| `GERP.RSsup4` | GERP conservation score (rejected substitutions) |
| `GTEx_eQTL_MaxCPP` | Max causal probability for GTEx eQTLs |
| `H3K27ac_Hnisz` | Active enhancer/promoter histone mark (Hnisz) |
| `H3K27ac_PGC2` | H3K27ac mark from PGC2 consortium |
| `H3K4me1_Trynka` | Enhancer histone mark (H3K4me1, Trynka) |
| `H3K4me3_Trynka` | Promoter histone mark (H3K4me3, Trynka) |
| `H3K9ac_Trynka` | Active promoter mark (H3K9ac, Trynka) |
| `Human_Enhancer_Villar` | Enhancer region from Villar et al. |
| `Human_Enhancer_Villar_Species_Enhancer_Count` | Cross-species enhancer conservation count |
| `Human_Promoter_Villar` | Promoter region from Villar et al. |
| `Human_Promoter_Villar_ExAC` | Constraint of Villar promoters using ExAC data |
| `Intron_UCSC` | Is the variant in an intron region |
| `MAF_Adj_ASMC` | Allele age from ASMC model (adjusted for MAF) |
| `MAF_Adj_LLD_AFR` | LD level in African populations (MAF adjusted) |
| `MAF_Adj_Predicted_Allele_Age` | Predicted allele age (adjusted for MAF) |
| `Nucleotide_Diversity_10kb` | Genetic diversity in a 10kb window |
| `Promoter_UCSC` | Is the variant in a promoter region |
| `Recomb_Rate_10kb` | Recombination rate in a 10kb window |
| `Repressed_Hoffman` | Repressed chromatin annotation (Hoffman) |
| `SNP` | SNP name or rsID |
| `SuperEnhancer_Hnisz` | Super-enhancer annotation (Hnisz) |
| `TFBS_ENCODE` | Transcription factor binding site (ENCODE) |
| `TSS_Hoffman` | Transcription start site (Hoffman) |
| `UTR_3_UCSC` | 3' untranslated region from UCSC |
| `UTR_5_UCSC` | 5' untranslated region from UCSC |
| `WeakEnhancer_Hoffman` | Weak enhancer regions (Hoffman) |
| `alt` | The variant DNA base |
| `base` | The original DNA base in the reference genome |
| `chrom` | Which chromosome the variant is on |
| `consequence` | The effect of the variant (e.g. intronic, UTR) |
| `genetic_dist` | Genetic distance in cM for the SNP |
| `label` | 1 if causal, 0 if not |
| `ld_score` | How correlated the variant is with nearby ones |
| `maf` | How common the less frequent version of the variant is |
| `non_synonymous` | Non-synonymous variant (changes protein) |
| `pip` | Statistical confidence that the variant is causal |
| `pos` | The position of the variant on the chromosome |
| `ref` | The original DNA base in the reference genome |
| `synonymous` | Synonymous variant (does not change protein) |
| `trait` | Trait or disease this variant is linked to |

## Glossary of Genomic Annotation Terms

| Term | Definition |
|------|------------|
| `.flanking.500` or `.500` | Indicates that a variant lies within **500 base pairs** of a genomic feature, such as a promoter or enhancer. Used to capture nearby regulatory influence. |
| Flanking | A region immediately **upstream or downstream** of a specific feature (e.g., TSS, enhancer). Flanking regions may still affect gene regulation. |
| Enhancer | A **regulatory DNA sequence** that increases the transcription of a target gene, often functioning at a distance from the gene it regulates. |
| Promoter | A **DNA region immediately upstream of a gene's transcription start site (TSS)**. It acts as a docking site for RNA polymerase and transcription factors to initiate transcription. |
| Super-enhancer | A **large cluster of enhancers** that drive high expression of genes, often involved in cell identity or disease. Identified by very high levels of enhancer-associated marks like H3K27ac. |
| TSS (Transcription Start Site) | The genomic position where transcription of a gene begins. |
| CTCF | A **transcription factor** that plays a key role in chromatin looping and genome organization. CTCF sites often mark boundaries of regulatory domains. |
| DHS (DNase I Hypersensitive Site) | Genomic regions with **open chromatin** that are accessible to transcription factors. Identified via DNase I digestion. |
| QTL (Quantitative Trait Locus) | A genomic region that influences a quantitative trait. e.g., **eQTL** affects gene expression; **mQTL** affects methylation. |
| H3K27ac, H3K4me1, etc. | **Histone modifications** associated with active enhancers or promoters. For example, **H3K27ac** is a mark of active enhancers and promoters. |
| phastCons | A **conservation score** indicating how evolutionarily conserved a nucleotide is across multiple species. High scores imply functional importance. |
| GERP | Another conservation score, measuring **evolutionary constraint**. High GERP scores indicate fewer substitutions than expected under neutrality. |
| CpG Content | The proportion of **Cytosine-phosphate-Guanine** dinucleotides. CpG-rich regions are often promoters and regulatory regions. |
| Allele Age | An estimate of how **old** a genetic variant is in evolutionary time, often inferred from coalescent models or linkage patterns. |
| LD Score | A measure of **linkage disequilibrium**, indicating how strongly a variant is correlated with nearby variants. |
| ASMC | A model for estimating **allele age** using ancestral recombination graphs. Frequently used in large-scale human genetics. |
| Background Selection | A form of **purifying selection** that removes deleterious alleles and reduces variation in linked regions. |
| Nucleotide Diversity | A measure of the **genetic variation** in a population at the nucleotide level. Higher values imply more diversity. |

