# Title Recommendation for Indonesian News Article Using Long Short-Term Memory with Attention Mechanism
### by Joshia Cahyadi - 10119086

## Welcome to the repository! 👋
This repository serves as the hub for Joshia's undergraduate thesis code. This project focuses on creating a deep learning model to recommend titles for Indonesian news articles using Long Short-Term Memory with Attention Mechanism.

## Features

- 🤖 LSTM with Attention Mechanism
- 🌐 Support for Indonesian Text
- 📊 Model Evaluation Metrics using BERTScore

## Table of Contents
- [Introduction](#Introduction)
- [Results](#Results)

## Introduction
News articles are one of the most widely circulated sources of information on the internet. The existence of news articles makes people more sensitive toward the situation
happening in the real world. Title, in this case, certainly has an important role to give some descriptions of an article. Unfortunately, nowadays many titles are made very
interesting, but not following the content of the corresponding news article. 

This final project focuses on developing a title recommendation program specifically tailored 
for Indonesian news articles using a deep learning approach. The developed title recommendation program in this final project leverages the sequence-to-sequence
architecture with long short-term memory (LSTM) as the basis. Additionally, this architecture also incorporates an attention mechanism to help the model to capture 
important information inside the news article.

## Results
The attention mechanism model is trained using 300,000 pairs of article and news title data. Training this model takes 
approximately 3 hours and 25 minutes. There are two results obtained for the attention mechanism model, namely the loss value obtained during the model training process 
and the BERTScore obtained after the model training process is complete. The loss value and BERTScore for the model with attention mechanism can be seen in 
Figures I and II.

<p align="center">
  <img src="https://private-user-images.githubusercontent.com/70884538/288792148-ec9f1a17-c310-422a-b943-c92b0baab7b6.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTEiLCJleHAiOjE3MDE5NjA5NTQsIm5iZiI6MTcwMTk2MDY1NCwicGF0aCI6Ii83MDg4NDUzOC8yODg3OTIxNDgtZWM5ZjFhMTctYzMxMC00MjJhLWI5NDMtYzkyYjBiYWFiN2I2LnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFJV05KWUFYNENTVkVINTNBJTJGMjAyMzEyMDclMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjMxMjA3VDE0NTA1NFomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTJmMWUwMjJiZjBhMDc1MjBiM2E0MDFhMGIyN2M4NTc3Yjk5ODk2YTViY2I0NzU1NTBkNWI4ZGY3MmQxYTZjMmQmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0JmFjdG9yX2lkPTAma2V5X2lkPTAmcmVwb19pZD0wIn0.rXf-rh56BLxneX_uEqGIUHAlSXMVzwC6gPFyKv4FQiY" alt="Figure I">
</p>
<h3 align="center">Figure I - Loss Value for Attention Mechanism Model</h3>
<br><br><br>
<p align="center">
  <img src="https://private-user-images.githubusercontent.com/70884538/288792140-5351d5b7-a521-4bfc-b1b5-7c93c792de76.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTEiLCJleHAiOjE3MDE5NjAyOTAsIm5iZiI6MTcwMTk1OTk5MCwicGF0aCI6Ii83MDg4NDUzOC8yODg3OTIxNDAtNTM1MWQ1YjctYTUyMS00YmZjLWIxYjUtN2M5M2M3OTJkZTc2LnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFJV05KWUFYNENTVkVINTNBJTJGMjAyMzEyMDclMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjMxMjA3VDE0Mzk1MFomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWM5OTk1NDVmMTFhMDUwYmUwMmE4N2ViZjIzZGJjNzhjZDhhMDRiMjI2ZWM1OGQzYWNjYzUzNTQxNmZmZTdiYTQmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0JmFjdG9yX2lkPTAma2V5X2lkPTAmcmVwb19pZD0wIn0.iltswla12IqJiwRraH9jTqwhIHaMah1JZ6rYDuL9azI" alt="Figure II">
</p>
<h3 align="center">Figure II - BERTScore Distribution for Attention Mechanism Model</h3>
