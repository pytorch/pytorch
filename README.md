# Auto-Improve AI System

This project is an "auto-improve system" consisting of AI agents that collaboratively enhance a shared code library and build machine learning models.

## Overview

The system has two main types of agents:

*   **Framework Improve Agents:** These agents analyze the `shared_library` and propose improvements to its structure, algorithms, or introduce entirely new concepts for model building. They use a benchmarking system to validate their improvements.
*   **Model Builder Agents:** These agents use the `shared_library` to construct, train, and evaluate machine learning models based on user-provided datasets.

## Core Components

*   `shared_library/`: Contains the core code modules used by agents.
*   `agents/`: Houses the logic for `FrameworkImproveAgent` and `ModelBuilderAgent`.
*   `benchmarking/`: System for evaluating improvements made by Framework Improve Agents.
*   `evaluation/`: System for evaluating models built by Model Builder Agents.
*   `config/`: Configuration files for the system (e.g., API keys, agent settings).
*   `data/`: Storage for datasets and model outputs.
*   `main.py`: The main Command Line Interface (CLI) for interacting with the system.

## Setup (Placeholder)

Detailed setup instructions will be added here.

## Usage (Placeholder)

Instructions on how to use the system via `main.py` will be added here.
