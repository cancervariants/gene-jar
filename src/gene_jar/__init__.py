"""Gene JAR: a tool for harmonizing ambiguous gene symbols."""

from importlib.metadata import version as _version

from gene_jar.resolver import GeneJar, MatchType, SearchColumn

__all__ = ["GeneJar", "MatchType", "SearchColumn"]
__version__ = _version("gene-jar")
