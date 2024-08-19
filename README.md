# IT Service Ticket Classification using DistilBERT

This project focuses on classifying IT service tickets into predefined categories using a DistilBERT model. The dataset comprises IT service tickets labeled with various categories, and the goal is to train a machine learning model to accurately classify new tickets.

## Table of Contents
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Usage](#usage)
- [License](#license)

## Dataset

The dataset contains 47,837 IT service tickets with two columns:

- **text**: The content of the IT service ticket.
- **label**: The category to which the ticket belongs.

The categories include:
- `Hardware`
- `HR Support`
- `Access`
- `Miscellaneous`
- `Storage`
- `Purchase`
- `Internal Project`
- `Administrative rights`

The dataset is split into:

- **Training Set**: 80% of the data
- **Validation Set**: 10% of the data
- **Test Set**: 10% of the data

## Project Structure

