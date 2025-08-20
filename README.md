# ML-Futbol ⚽️🤖

*A reinforcement-learning framework that learns how to exploit an opponent’s weaknesses and outputs match-specific tactical game-plans.*

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)  

---

## 📑 Table of Contents
1. [Overview](#overview)
2. [Roadmap](#roadmap)
3. [Contributing](#contributing)
4. [Documentation](#documentation)
5. [License](#license)

---

## Overview
**RL-Tactics** helps coaching staffs answer the question ➡️ *“What is the best way to line up and press against Team X next Saturday?”*  
It does so by training a **reinforcement-learning (RL)** agent in a simulation environment and surfacing match-specific tactical plans ranked by expected goal differential (**xGD**).
---

## Roadmap
- **Version 0 (current)** — we’re at **v0**; core research sandbox with continuous improvements underway.  
- **Version 0.1** — first prototype of the RL model that can generate basic tactic recommendations.  
- **Version 1.0** — final public release with refined model, full documentation, and a polished web interface.  

---

## Contributing
1. **Fork** the repo & create your feature branch (`git checkout -b feat/awesome-idea`).  
2. **Commit** using [Conventional Commits](https://www.conventionalcommits.org/) & open a PR.  
3. Ensure `pytest` & `pre-commit` hooks pass.  

All contributors must follow our **Code of Conduct**.

---

## Documentation

Full documentation lives in the [`docs/`](docs/) folder:

- [Getting Started](docs/getting_started.md)
- [Usage Guide](docs/usage.md)
- [Architecture](docs/architecture.md)
- [Pitch Modeling](docs/pitch_modeling.md)
- [Team Classifier](docs/team_classifier.md)
- [Model Files](docs/models.md)
- [Output Files](docs/outputs.md)
- [Data Format](docs/data_format.md)
- [Developer Guide](docs/developer_guide.md)

---

## License
This project is licensed under the **MIT License** — see `LICENSE` for details.

