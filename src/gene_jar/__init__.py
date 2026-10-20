"""Gene JAR: a tool for harmonizing ambiguous gene symbols."""

from importlib.metadata import PackageNotFoundError, version

from gene_jar.resolver import GeneJar, MatchType, SearchColumn, SymbolCategory

__all__ = ["GeneJar", "MatchType", "SearchColumn", "SymbolCategory", "__version__"]


try:
    __version__ = version("gene_jar")
except PackageNotFoundError:
    __version__ = "unknown"
finally:
    del version, PackageNotFoundError
