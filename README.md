# 🎶 Speech and Music Separation using Non-negative Matrix Factorization (NMF)

**Author:** Utkarsh Rana 
**Date:** October 2025  
**Based on:**  
> *Non-negative Matrix Factorization for Speech/Music Separation using Source-dependent Decomposition Rank, Temporal Continuity Term, and Filtering*  
> **S. Abdali, B. NaserSharif**, Biomedical Signal Processing and Control, Vol. 36, pp. 168–175, 2017.  

---

## 🧭 Overview

This repository implements **speech and music source separation** from a single-channel audio mixture using **Non-negative Matrix Factorization (NMF)** and **Wiener filtering**.

The project explores and reproduces the method described in Abdali & NaserSharif (2017), extending the standard NMF by incorporating:

- **Source-dependent decomposition ranks**
- **Temporal continuity regularization**
- **Spectral masking (Wiener filter) post-processing**

This approach enables effective separation of speech and music in mixed audio signals.

---

## Contents

| File | Description |
|------|--------------|
| `nmf4_wiener.py` | Main Python implementation for NMF-based separation |
| `nmf_project_report.pdf` | Detailed report explaining theory, code, and results |
| `Non_negative_matrix_factorization_for_sp.pdf` | Original reference research paper |
| `data/` | Folder containing sample speech, music, and mixture audio files |
| `output/` | Folder where separated results are saved |

---

## 🎧 Problem Description

Single-channel **source separation** aims to extract individual sources (speech, music) from one mixed signal without access to isolated recordings.



