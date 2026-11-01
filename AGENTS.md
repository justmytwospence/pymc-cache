# Repository guidance

This repository owns cache identity, storage, and policy only. It must not depend on cloudposterior, pymc-monitor or Modal. Keep `.extraction-source/` temporary, ignored and out of all artifacts. Follow `CONTRIBUTING.md`; preserve old external cache data. Never silently reuse an entry when a model or sampler cannot be identified safely.
