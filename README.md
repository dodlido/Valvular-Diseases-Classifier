# Valvular-Diseases-Classifier

This repository contains a binary valvular-diseases classification system.
  
  Input: a recording of heart sounds, acquired usually by a stetohscope.
  Output: 1/0 depending if the patient is healthy or sick

The system is composed of 2 main stages:
  1. Segmentation - division of a recording into different heart cycles, starting at S1 sounds.
  2. Classification - classification of each heart cycle, and of a complete recording via majority vote.

For further information, please refer to the project book in the repo.
