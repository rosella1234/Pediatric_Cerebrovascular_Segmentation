# CaravelMetrics attribution

The modules in `src/tof_mra/_caravel/` originate from the CaravelMetrics study scripts supplied by the user and the public [i-vesseg/CaravelMetrics](https://github.com/i-vesseg/CaravelMetrics) project. This repository is a study companion and does not claim authorship of that framework.

Upstream authors include Daniele Falcetta, Liane S. Canas, Lorenzo Suppa, Matteo Pentassuglia, Jon Cleary, Marc Modat, Sébastien Ourselin, and Maria A. Zuluaga. Please cite the CaravelMetrics paper and acknowledge its project when using those methods. The upstream README describes its licensing as MIT; this notice preserves attribution and does not assign a new license to third-party work.

The public source was checked at commit `98482a029bd15d041466024eb741770a4f6f48d5`. Metric_extractor and Atlas_Registration in the uploaded archive match the upstream Python AST. The uploaded Graph_extractor differs only by passing `affine.tolist()` to the Vedo mesh transform; that version is preserved.

Packaging changes:

- Module filenames become graph.py, metrics.py and registration.py inside a private subpackage.
- Registration explicitly directs crop output to the subject directory, avoiding an implicit working-directory file and move step. Numerical registration parameters are unchanged.
- New orchestration replaces hard-coded paths, implicit file matching and notebook state with configuration and manifest inputs.
- The wrapper shares the atlas transform across models and adds checked completion/resume records. Graph and metric calculations are retained.

The clinical metadata workbook, notebook output cells, server paths and original patient identifiers are not distributed. Source hashes and the upstream revision are recorded in SOURCE_PROVENANCE.json. Dependencies retain their own authorship, licensing and citation requirements.

