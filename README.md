# gene-jar

[![image](https://img.shields.io/pypi/v/gene-jar.svg)](https://pypi.python.org/pypi/gene-jar)
[![image](https://img.shields.io/pypi/l/gene-jar.svg)](https://pypi.python.org/pypi/gene-jar)
[![image](https://img.shields.io/pypi/pyversions/gene-jar.svg)](https://pypi.python.org/pypi/gene-jar)

[![Actions status](https://github.com/cancervariants/gene-jar/actions/workflows/checks.yaml/badge.svg)](https://github.com/cancervariants/gene-jar/actions/checks.yaml)

<!-- description -->
This resource is useful for resolving ambiguous gene symbols powered by thousands of annotations characterizing the relationships between genes and their symbols. Please see the [quickstart notebook](gene_jar_example_use.ipynb) for example usage.

Given a gene symbol, this package:
- finds all genes with this symbol as an alias or primary gene symbol
- provides relationships between the gene symbols and their genes, when available
- provides more detailed information when given the relationship type with the gene symbol

Relationship rank was determined using a calculated ambiguity score for each relationship category. See [gene jar resources repo](https://github.com/cancervariants/gene-jar-resources/tree/09ecaf9fdae49915a6fd35a6830b46c6ae182e2a/analysis/ranked_category_resolver)

### Annotation dataset sources

Gene JAR separates annotation dataset generation from dataset consumption.

The [gene jar resources repo](https://github.com/cancervariants/gene-jar-resources) repository generates the annotation datasets used by Gene JAR. The generated datasets are published as versioned archives on [Zenodo](https://zenodo.org/records/23965261).

The `gene-jar` package provides the Python API for accessing these datasets. Its dataset loader retrieves the latest published Zenodo version, validates the downloaded archive against its published checksum, and caches the extracted files locally.

This separation allows annotation datasets to be updated independently of the `gene-jar` Python package.

## Dataset Sources and Versions

Gene JAR separates annotation dataset generation from dataset consumption.

The [gene-jar-analysis](https://github.com/cancervariants/gene-jar-analysis) repository generates the gene annotation datasets used by Gene JAR. These datasets are published on [Zenodo](https://zenodo.org/records/23965261), allowing annotation data to be updated independently of the `gene-jar` Python package.

The `gene-jar` package provides the Python API for accessing these datasets. It downloads, validates, and caches the published annotation files locally.

### Downloading Datasets

By default, `GeneJar.download_datasets()` downloads the latest published dataset from Zenodo:

```python
from gene_jar import GeneJar

jar = GeneJar.download_datasets()
```

For reproducible analyses, a specific dataset release can be selected using its **Zenodo record ID**:

```python
jar = GeneJar.download_datasets(version="23965261")
```

Each published Zenodo version receives a unique record ID. The loader verifies that the selected record belongs to the Gene JAR dataset collection.

**Note:** The `version` parameter accepts a Zenodo record ID, not a semantic version string such as `"1.0.0"`.

### Dataset Caching

Downloaded datasets are stored locally under:

```text
~/.cache/gene-jar/<record-id>/
```

The loader validates the downloaded archive against its published Zenodo checksum and extracts the required annotation files. Previously downloaded datasets are reused when the cached archive and extracted files pass the loader's validation checks.

Newly published dataset versions can be retrieved without updating or reinstalling the `gene-jar` Python package.

Resources used for annotations include:
[Ensembl](https://www.ensembl.org/index.html)
[HGNC](https://www.genenames.org/)
[NCBI Gene](https://www.ncbi.nlm.nih.gov/gene/)
[Saccharomyces Genome Database](http://sgd-archive.yeastgenome.org/)
[FLJ Human cDNA Database](https://flj.lifesciencedb.jp/top/sys_info/02_about_database/accession_no/download_v032.html)
[OMIM](https://omim.org/downloads/)
[Uniprot](https://www.uniprot.org/)
[Mouse Genome Informatics](https://www.informatics.jax.org/)
<!-- /description -->

---

## Installation

Install from [PyPI](https://pypi.org/project/gene-jar/):

```shell
python3 -m pip install gene-jar
```

---

## Feedback and contributing

We welcome bug reports, feature requests, and code contributions from users and interested collaborators. Please make an issue.
