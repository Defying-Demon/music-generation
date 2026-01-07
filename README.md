# Music Generation via Multi-Stream LSTM Networks

## Abstract

This project presents a deep learning framework for algorithmic music composition, specifically focusing on the generation of symbolic music sequences. By leveraging Long Short-Term Memory (LSTM) recurrent neural networks, the model captures long-range temporal dependencies inherent in musical structures. The system is designed to learn the statistical properties of pitch and rhythm from a corpus of MIDI files and generate novel, coherent musical compositions that maintain both melodic continuity and rhythmic variety.

## Problem Formulation

Music generation can be modeled as a sequence modeling problem where the goal is to predict the next musical event given a history of interconnected events. Unlike text generation, musical events are multi-dimensional, primarily consisting of **Pitch** (the frequency of the note) and **Duration** (the temporal length of the note). This project addresses the challenge of jointly modeling these two attributes to produce expressive monophonic melodies.

## Model Architecture

The proposed architecture utilizes a multi-input, multi-output LSTM network designed to process pitch and duration as distinct but correlated features.

### 1. Input Representation
The model accepts two parallel input sequences:
- **Pitch Sequence**: A sequence of discrete tokens representing musical notes (e.g., "C4", "G#5") or chords.
- **Duration Sequence**: A sequence of discrete tokens representing the length of each note (e.g., "0.5", "1.0").

### 2. Feature Extraction
*   **Embeddings**: discrete inputs are mapped to continuous vector spaces to capture semantic relationships between notes and rhythmic values.
    *   Pitch Embedding Dimension: $d_{pitch} = 128$
    *   Duration Embedding Dimension: $d_{duration} = 32$
*   **Fusion**: The separate embeddings are concatenated to form a unified feature vector for each time step.
*   **Regularization**: A **Spatial Dropout (1D)** mechanism is applied to the fused embeddings to prevent co-adaptation of feature maps, followed by **Layer Normalization** to stabilize training dynamics.

### 3. Recurrent Layers
The core temporal processing is performed by a stack of three LSTM layers:
1.  **LSTM Layer 1**: 256 units, return sequences.
2.  **LSTM Layer 2**: 128 units, return sequences.
3.  **LSTM Layer 3**: 128 units, processing the sequence to extract a final context vector.

### 4. Output Heads
The context vector from the final LSTM layer is passed through a shared Dense layer ($d=128$) with ReLU activation. This feeds into two independent output heads:
*   **Pitch Head**: A Softmax classifier predicting the probability distribution over the pitch vocabulary.
*   **Duration Head**: A Softmax classifier predicting the probability distribution over the duration vocabulary.

## Code for Music Generation

* Author: [Realsanjeev](https://realsanjeev.github.io/)

**This code was originally developed as part of my undergraduate Major Project.** The core implementation remains unchanged; however, it has been updated to work with newer versions of the required libraries. Additionally, some preprocessing steps were shortened to ensure quick run of the notebook.

## Dataset
The dataset used was [ESAC](https://www.esac-data.org/). The dataset contains 5K+ MIDI files.

### Generation
The script automatically enters generation mode after training.
