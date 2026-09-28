"""Utilities for resolving gene symbols across multiple harmonized reference datasets."""

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, ClassVar

import pandas as pd


class MatchType(StrEnum):
    """Match options"""

    IDENTICAL = "identical"
    PARTIAL = "partial"


@dataclass
class GeneMatch:
    """A single resolved gene and its associated identifiers."""

    primary_gene_symbol: str
    hgnc_id: set[str]
    ncbi_id: set[str]
    ensg_id: set[str]


@dataclass
class AmbiguityResult:
    """Result of a gene symbol ambiguity check."""

    is_ambiguous: bool
    gene_matches: list[GeneMatch]


class GeneJar:
    """Resolve gene symbols across multiple harmonized gene reference datasets.

    :param primary_df: DataFrame containing primary gene symbols. Must include
        ``gene_symbol`` and ``primary_gene_symbol``.
    :param ortholog_df: DataFrame containing ortholog symbols.
    :param flj_clone_df: DataFrame containing FLJ clone symbols.
    :param phenotype_df: DataFrame containing phenotype-related prefix symbols.
    :param hgnc_gene_group_df: DataFrame containing HGNC gene group symbols.
    :param gene_id_df: DataFrame containing gene identifier symbols.
    :param mgi_withdrawn_df: DataFrame containing withdrawn MGI symbols.
    :param ncbi_related_gene_df: DataFrame containing NCBI gene records that are related.
    :param ncbi_gene_interaction_df: DataFrame containing NCBI gene interaction symbols.
    :param ncbi_gene_neighbor_df: DataFrame containing NCBI gene neighbor symbols.
    :param placeholder_df: DataFrame containing placeholder symbols.
    :param previous_df: DataFrame containing previous gene symbols.
    :param protein_mass_df: DataFrame containing protein mass symbols.
    :param alternate_abbreviation_df: DataFrame containing alternate abbreviation symbols
    """

    RANK_ORDER: ClassVar[list[str]] = [
        "Primary Gene Symbol",
        "Previous Symbol",
        "Clone Name Symbol",
        "Gene Identifier Symbol",
        "Placeholder Symbol",
        "Ortholog Symbol",
        "Alternate Abbreviation Symbol",
        "Withdrawn Ortholog Symbol",
        "Phenotype Symbol",
        "Gene Group Symbol",
        "Protein Mass Symbol",
        "Related Gene Symbol",
        "Gene Neighbor Symbol",
        "Gene Interaction Symbol",
    ]

    def __init__(
        self,
        *,
        primary_df: pd.DataFrame,
        ortholog_df: pd.DataFrame,
        flj_clone_df: pd.DataFrame,
        phenotype_df: pd.DataFrame,
        hgnc_gene_group_df: pd.DataFrame,
        gene_id_df: pd.DataFrame,
        mgi_withdrawn_df: pd.DataFrame,
        ncbi_related_gene_df: pd.DataFrame,
        ncbi_gene_interaction_df: pd.DataFrame,
        ncbi_gene_neighbor_df: pd.DataFrame,
        placeholder_df: pd.DataFrame,
        previous_df: pd.DataFrame,
        protein_mass_df: pd.DataFrame,
        alternate_abbreviation_df: pd.DataFrame,
    ):
        """Initialize with dataframes.
        primary_df and ortholog_df must have columns: 'gene_symbol', 'primary_gene_symbol'
        """
        self.dfs = {
            "Primary Gene Symbol": primary_df,
            "Ortholog Symbol": ortholog_df,
            "Clone Name Symbol": flj_clone_df,
            "Phenotype Symbol": phenotype_df,
            "Gene Group Symbol": hgnc_gene_group_df,
            "Gene Identifier Symbol": gene_id_df,
            "Withdrawn Ortholog Symbol": mgi_withdrawn_df,
            "Related Gene Symbol": ncbi_related_gene_df,
            "Gene Interaction Symbol": ncbi_gene_interaction_df,
            "Gene Neighbor Symbol": ncbi_gene_neighbor_df,
            "Placeholder Symbol": placeholder_df,
            "Previous Symbol": previous_df,
            "Protein Mass Symbol": protein_mass_df,
            "Alternate Abbreviation Symbol": alternate_abbreviation_df,
        }

        self.column_map = {
            "Primary Gene Symbol": ("primary_gene_symbol"),
        }

        self.qualifier_map = {
            "Ortholog Symbol": "Matching Species",
            "Phenotype Symbol": "Matching Phenotype Symbol",
            "Gene Group Symbol": "Matching Abbreviation",
            "Gene Identifier Symbol": "Identifier Match Source",
            "Related Gene Symbol": "Relationship",
            "Gene Neighbor Symbol": "neighbor_gene_type",
            "Placeholder Symbol": "Placeholder Symbol Match Type",
            "Previous Symbol": "Previous Symbol Source",
        }

        for category in self.dfs:
            self.column_map.setdefault(category, "alias_symbol")

        self.rank_map = {
            category: rank for rank, category in enumerate(self.RANK_ORDER)
        }

    def symbol_categories(self) -> list[str]:
        """Return the valid category names accepted by resolve()."""
        return sorted(self.dfs)

    @staticmethod
    def _identifier_set(
        group: pd.DataFrame,
        column: str,
    ) -> set[str]:
        """Return the unique non-null identifiers in a candidate group.

        :param group: Rows associated with one candidate gene.
        :param column: Identifier column to combine.
        :return: Unique identifiers represented as strings.
        """
        if column not in group:
            return set()

        identifiers = set()

        for value in group[column].dropna():
            if isinstance(value, (set, list, tuple)):
                identifiers.update(
                    str(identifier) for identifier in value if pd.notna(identifier)
                )
            else:
                identifiers.add(str(value))

        return identifiers

    @staticmethod
    def _has_qualifier(value: str | list[str]) -> bool:
        """Return whether a qualifier value is non-null and non-empty."""
        if isinstance(value, list):
            return len(value) > 0

        return pd.notna(value)

    def lookup(
        self,
        symbol: str,
        symbol_category: str = "Primary Gene Symbol",
        match_type: MatchType = MatchType.IDENTICAL,
    ) -> pd.DataFrame:
        """Lookup a gene symbol from a specific symbol category.

        Searches both the symbol and primary gene symbol columns.

        :param symbol: Gene symbol to search for.
        :param symbol_category: Symbol category to search.
        :param match_type: Type of matching to perform.
        :return: Matching rows.
        """
        if symbol_category not in self.dfs:
            message = (
                f"Unknown symbol category '{symbol_category}'. "
                f"Available: {sorted(self.dfs)}"
            )
            raise ValueError(message)

        match_type = MatchType(match_type)
        df = self.dfs[symbol_category]
        target = symbol.casefold()

        if symbol_category == "Primary Gene Symbol":
            symbol_column = "gene_symbol"
        else:
            symbol_column = "alias_symbol"

        symbols = df[symbol_column].astype("string").str.casefold()
        primary_symbols = df["primary_gene_symbol"].astype("string").str.casefold()

        if match_type is MatchType.IDENTICAL:
            mask = symbols.eq(target).fillna(False) | primary_symbols.eq(target).fillna(
                False
            )
        else:
            mask = symbols.str.contains(
                target, regex=False, na=False
            ) | primary_symbols.str.contains(target, regex=False, na=False)

        return df.loc[mask].copy()

    def resolve(
        self,
        symbol: str,
        match_type: MatchType = MatchType.IDENTICAL,
    ) -> pd.DataFrame:
        """Resolve a symbol and return ranked candidate genes.

        Each relationship DataFrame is searched independently. The relationship
        category represented by the DataFrame is assigned to each matching
        candidate gene. Candidates are ranked using their highest-priority matched
        relationship category. When candidates share the same highest-priority
        category, the candidate with more matched relationship categories is
        ranked first.

        :param symbol: Gene symbol to resolve.
        :param match_type: Type of symbol matching to perform.
        :return: Ranked candidate genes and their associated identifiers.
        """
        match_type = MatchType(match_type)
        target = symbol.casefold()
        matches = []

        for category, df in self.dfs.items():
            symbol_column = self.column_map[category]

            normalized_symbols = df[symbol_column].astype("string").str.casefold()

            if match_type is MatchType.IDENTICAL:
                mask = normalized_symbols.eq(target).fillna(False)
            else:
                mask = normalized_symbols.str.contains(
                    target,
                    regex=False,
                    na=False,
                )

            category_matches = df.loc[mask].copy()

            if category_matches.empty:
                continue

            # Record exactly why this row matched.
            category_matches["matched_category"] = category
            category_matches["matched_symbol"] = category_matches[symbol_column]

            qualifier_column = self.qualifier_map.get(category)

            if qualifier_column is not None:
                category_matches["qualifier"] = [
                    {qualifier_column: value} if self._has_qualifier(value) else {}
                    for value in category_matches[qualifier_column]
                ]
            else:
                category_matches["qualifier"] = [
                    {} for _ in range(len(category_matches))
                ]

            matches.append(category_matches)

        if not matches:
            return pd.DataFrame(
                columns=[
                    "candidate_rank",
                    "primary_gene_symbol",
                    "HGNC_ID",
                    "NCBI_ID",
                    "ENSG_ID",
                    "best_category",
                    "qualifier",
                    "matched_categories",
                    "relationship_count",
                ]
            )

        all_matches = pd.concat(
            matches,
            ignore_index=True,
        )

        candidates = []

        for primary_symbol, group in all_matches.groupby(
            "primary_gene_symbol",
            sort=False,
            dropna=False,
        ):
            matched_categories = sorted(
                group["matched_category"].dropna().unique(),
                key=lambda category: self.rank_map[category],
            )

            best_category = matched_categories[0]

            best_category_rows = group[group["matched_category"] == best_category]

            qualifier = {}

            for value in best_category_rows["qualifier"]:
                qualifier.update(value)

            matched_symbols = set(group["matched_symbol"].dropna().astype(str))

            candidates.append(
                {
                    "primary_gene_symbol": primary_symbol,
                    "HGNC_ID": self._identifier_set(group, "HGNC_ID"),
                    "NCBI_ID": self._identifier_set(group, "NCBI_ID"),
                    "ENSG_ID": self._identifier_set(group, "ENSG_ID"),
                    "best_category": best_category,
                    "qualifier": qualifier,
                    "matched_symbol": matched_symbols,
                    "matched_categories": matched_categories,
                    "relationship_count": len(matched_categories),
                    "_category_rank": self.rank_map[best_category],
                }
            )

        result = pd.DataFrame(candidates)

        result = result.sort_values(
            by=[
                "_category_rank",
                "relationship_count",
                "primary_gene_symbol",
            ],
            ascending=[
                True,
                False,
                True,
            ],
            kind="stable",
        ).reset_index(drop=True)

        result.insert(
            0,
            "candidate_rank",
            result.index + 1,
        )

        return result.drop(columns="_category_rank")

    def _flatten_unique(self, values: pd.Series) -> list[Any]:
        """Flatten nested identifier values and return unique non-null entries."""
        flattened = []

        for value in values.dropna():
            if isinstance(value, (set, list, tuple)):
                flattened.extend(value)
            else:
                flattened.append(value)

        return list(dict.fromkeys(flattened))

    def ambiguity_check(self, symbol: str) -> AmbiguityResult:
        """Check whether a symbol is associated with multiple primary genes."""
        gene_match = self.resolve(
            symbol=symbol,
            match_type=MatchType.IDENTICAL,
        )

        gene_matches = [
            GeneMatch(
                primary_gene_symbol=row["primary_gene_symbol"],
                hgnc_id=row["HGNC_ID"],
                ncbi_id=row["NCBI_ID"],
                ensg_id=row["ENSG_ID"],
            )
            for _, row in gene_match.iterrows()
        ]

        primary_symbols = {gene.primary_gene_symbol for gene in gene_matches}

        return AmbiguityResult(
            is_ambiguous=len(primary_symbols) > 1,
            gene_matches=gene_matches,
        )
