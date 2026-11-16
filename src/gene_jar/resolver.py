"""Utilities for resolving gene symbols across multiple harmonized reference datasets."""

import ast
from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar, Literal

import pandas as pd


class MatchType(StrEnum):
    """Match options."""

    IDENTICAL = "identical"
    PARTIAL = "partial"


class SearchColumn(StrEnum):
    """Gene symbol columns to search."""

    BOTH = "both"
    ALIAS = "gene_symbol"
    PRIMARY = "primary_gene_symbol"


class SymbolCategory(StrEnum):
    """Gene symbol relationship categories."""

    PRIMARY = "Primary Gene Symbol"
    ORTHOLOG = "Ortholog Symbol"
    CLONE_NAME = "Clone Name Symbol"
    PHENOTYPE = "Phenotype Symbol"
    GENE_GROUP = "Gene Group Symbol"
    GENE_IDENTIFIER = "Gene Identifier Symbol"
    WITHDRAWN_ORTHOLOG = "Withdrawn Ortholog Symbol"
    RELATED_GENE = "Related Gene Symbol"
    GENE_INTERACTION = "Gene Interaction Symbol"
    GENE_NEIGHBOR = "Gene Neighbor Symbol"
    PLACEHOLDER = "Placeholder Symbol"
    PREVIOUS = "Previous Symbol"
    PROTEIN_MASS = "Protein Mass Symbol"
    ALTERNATE_ABBREVIATION = "Alternate Abbreviation Symbol"


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

    ``context`` changes ranking priorities across all relationship categories.
    ``strict_context`` restricts which relationship categories are searched.

    These parameters are mutually exclusive.
    """

    RANK_ORDER: ClassVar[list[SymbolCategory]] = [
        SymbolCategory.PRIMARY,
        SymbolCategory.GENE_IDENTIFIER,
        SymbolCategory.WITHDRAWN_ORTHOLOG,
        SymbolCategory.CLONE_NAME,
        SymbolCategory.PHENOTYPE,
        SymbolCategory.PLACEHOLDER,
        SymbolCategory.PREVIOUS,
        SymbolCategory.ALTERNATE_ABBREVIATION,
        SymbolCategory.ORTHOLOG,
        SymbolCategory.GENE_GROUP,
        SymbolCategory.RELATED_GENE,
        SymbolCategory.PROTEIN_MASS,
        SymbolCategory.GENE_INTERACTION,
        SymbolCategory.GENE_NEIGHBOR,
    ]

    CONTEXT_RANK_ORDERS: ClassVar[dict[str, list[SymbolCategory]]] = {
        "human": [
            SymbolCategory.PRIMARY,
            SymbolCategory.GENE_IDENTIFIER,
            SymbolCategory.PHENOTYPE,
            SymbolCategory.PREVIOUS,
            SymbolCategory.GENE_GROUP,
            SymbolCategory.WITHDRAWN_ORTHOLOG,
            SymbolCategory.CLONE_NAME,
            SymbolCategory.PLACEHOLDER,
            SymbolCategory.ALTERNATE_ABBREVIATION,
            SymbolCategory.ORTHOLOG,
            SymbolCategory.RELATED_GENE,
            SymbolCategory.PROTEIN_MASS,
            SymbolCategory.GENE_INTERACTION,
            SymbolCategory.GENE_NEIGHBOR,
        ],
        "non_human": [
            SymbolCategory.PRIMARY,
            SymbolCategory.GENE_IDENTIFIER,
            SymbolCategory.WITHDRAWN_ORTHOLOG,
            SymbolCategory.ORTHOLOG,
            SymbolCategory.CLONE_NAME,
            SymbolCategory.PHENOTYPE,
            SymbolCategory.PLACEHOLDER,
            SymbolCategory.PREVIOUS,
            SymbolCategory.ALTERNATE_ABBREVIATION,
            SymbolCategory.GENE_GROUP,
            SymbolCategory.RELATED_GENE,
            SymbolCategory.PROTEIN_MASS,
            SymbolCategory.GENE_INTERACTION,
            SymbolCategory.GENE_NEIGHBOR,
        ],
        "historical": [
            SymbolCategory.PRIMARY,
            SymbolCategory.GENE_IDENTIFIER,
            SymbolCategory.WITHDRAWN_ORTHOLOG,
            SymbolCategory.PREVIOUS,
            SymbolCategory.CLONE_NAME,
            SymbolCategory.PHENOTYPE,
            SymbolCategory.PLACEHOLDER,
            SymbolCategory.ALTERNATE_ABBREVIATION,
            SymbolCategory.ORTHOLOG,
            SymbolCategory.GENE_GROUP,
            SymbolCategory.RELATED_GENE,
            SymbolCategory.PROTEIN_MASS,
            SymbolCategory.GENE_INTERACTION,
            SymbolCategory.GENE_NEIGHBOR,
        ],
        "clinical": [
            SymbolCategory.PRIMARY,
            SymbolCategory.GENE_IDENTIFIER,
            SymbolCategory.PHENOTYPE,
            SymbolCategory.WITHDRAWN_ORTHOLOG,
            SymbolCategory.CLONE_NAME,
            SymbolCategory.PLACEHOLDER,
            SymbolCategory.PREVIOUS,
            SymbolCategory.ALTERNATE_ABBREVIATION,
            SymbolCategory.ORTHOLOG,
            SymbolCategory.GENE_GROUP,
            SymbolCategory.RELATED_GENE,
            SymbolCategory.PROTEIN_MASS,
            SymbolCategory.GENE_INTERACTION,
            SymbolCategory.GENE_NEIGHBOR,
        ],
        "protein": [
            SymbolCategory.PRIMARY,
            SymbolCategory.GENE_IDENTIFIER,
            SymbolCategory.PROTEIN_MASS,
            SymbolCategory.GENE_GROUP,
            SymbolCategory.WITHDRAWN_ORTHOLOG,
            SymbolCategory.CLONE_NAME,
            SymbolCategory.PHENOTYPE,
            SymbolCategory.PLACEHOLDER,
            SymbolCategory.PREVIOUS,
            SymbolCategory.ALTERNATE_ABBREVIATION,
            SymbolCategory.ORTHOLOG,
            SymbolCategory.RELATED_GENE,
            SymbolCategory.GENE_INTERACTION,
            SymbolCategory.GENE_NEIGHBOR,
        ],
    }

    STRICT_CONTEXT_GROUPS: ClassVar[dict[str, set[SymbolCategory]]] = {
        "ortholog": {
            SymbolCategory.ORTHOLOG,
            SymbolCategory.WITHDRAWN_ORTHOLOG,
        },
        "previous symbols": {
            SymbolCategory.PREVIOUS,
            SymbolCategory.WITHDRAWN_ORTHOLOG,
        },
    }

    SymbolCategoryOption = Literal[
        "Primary Gene Symbol",
        "Ortholog Symbol",
        "Clone Name Symbol",
        "Phenotype Symbol",
        "Gene Group Symbol",
        "Gene Identifier Symbol",
        "Withdrawn Ortholog Symbol",
        "Related Gene Symbol",
        "Gene Interaction Symbol",
        "Gene Neighbor Symbol",
        "Placeholder Symbol",
        "Previous Symbol",
        "Protein Mass Symbol",
        "Alternate Abbreviation Symbol",
    ]

    StrictContextOption = Literal[
        "ortholog",
        "previous symbols",
        "Primary Gene Symbol",
        "Ortholog Symbol",
        "Clone Name Symbol",
        "Phenotype Symbol",
        "Gene Group Symbol",
        "Gene Identifier Symbol",
        "Withdrawn Ortholog Symbol",
        "Related Gene Symbol",
        "Gene Interaction Symbol",
        "Gene Neighbor Symbol",
        "Placeholder Symbol",
        "Previous Symbol",
        "Protein Mass Symbol",
        "Alternate Abbreviation Symbol",
    ]

    ResolutionContext = Literal[
        "default",
        "human",
        "non_human",
        "historical",
        "clinical",
        "protein",
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
        """Initialize with harmonized relationship DataFrames."""
        self.dfs: dict[SymbolCategory, pd.DataFrame] = {
            SymbolCategory.PRIMARY: primary_df,
            SymbolCategory.ORTHOLOG: ortholog_df,
            SymbolCategory.CLONE_NAME: flj_clone_df,
            SymbolCategory.PHENOTYPE: phenotype_df,
            SymbolCategory.GENE_GROUP: hgnc_gene_group_df,
            SymbolCategory.GENE_IDENTIFIER: gene_id_df,
            SymbolCategory.WITHDRAWN_ORTHOLOG: mgi_withdrawn_df,
            SymbolCategory.RELATED_GENE: ncbi_related_gene_df,
            SymbolCategory.GENE_INTERACTION: ncbi_gene_interaction_df,
            SymbolCategory.GENE_NEIGHBOR: ncbi_gene_neighbor_df,
            SymbolCategory.PLACEHOLDER: placeholder_df,
            SymbolCategory.PREVIOUS: previous_df,
            SymbolCategory.PROTEIN_MASS: protein_mass_df,
            SymbolCategory.ALTERNATE_ABBREVIATION: alternate_abbreviation_df,
        }

        # Normalize identifiers across all reference datasets.
        # Copy each DataFrame to preserve the original input data.
        for category, df in self.dfs.items():
            normalized_df = df.copy()

            for column in ("HGNC_ID", "NCBI_ID", "ENSG_ID"):
                if column in normalized_df.columns:
                    normalized_df[column] = normalized_df[column].map(
                        self._parse_serialized_identifier_set
                    )

            self.dfs[category] = normalized_df

        self.column_map: dict[SymbolCategory, str] = {
            SymbolCategory.PRIMARY: "primary_gene_symbol",
        }

        self.qualifier_map: dict[SymbolCategory, str] = {
            SymbolCategory.ORTHOLOG: "Matching Species",
            SymbolCategory.PHENOTYPE: "Matching Phenotype Symbol",
            SymbolCategory.GENE_GROUP: "Matching Abbreviation",
            SymbolCategory.GENE_IDENTIFIER: "Identifier Match Source",
            SymbolCategory.RELATED_GENE: "Relationship",
            SymbolCategory.GENE_NEIGHBOR: "neighbor_gene_type",
            SymbolCategory.PLACEHOLDER: "Placeholder Symbol Match Type",
            SymbolCategory.PREVIOUS: "Previous Symbol Source",
        }

        for category in self.dfs:
            self.column_map.setdefault(category, "alias_symbol")

        self.rank_map: dict[SymbolCategory, int] = {
            category: rank for rank, category in enumerate(self.RANK_ORDER)
        }

    def symbol_categories(self) -> list[str]:
        """Return the valid individual gene symbol category names."""
        return sorted(category.value for category in SymbolCategory)

    @staticmethod
    def _parse_serialized_identifier_set(value: object) -> object:
        """Convert serialized Python sets into actual sets.

        Handles non-empty set representations such as "{'HGNC:5'}"
        and empty set representations such as "set()".

        Values that are not serialized sets are returned unchanged.
        """
        if isinstance(value, str):
            stripped = value.strip()

            if stripped == "set()":
                return set()

            if stripped.startswith("{"):
                try:
                    parsed = ast.literal_eval(stripped)

                    if isinstance(parsed, set):
                        return parsed

                except (ValueError, SyntaxError):
                    pass

        return value

    @staticmethod
    def _identifier_set(
        group: pd.DataFrame,
        column: str,
    ) -> set[str]:
        """Return unique non-null identifiers for a candidate gene."""
        if column not in group:
            return set()

        identifiers: set[str] = set()

        for value in group[column].dropna():
            if isinstance(value, (set, list, tuple)):
                identifiers.update(
                    str(identifier) for identifier in value if pd.notna(identifier)
                )
            else:
                identifiers.add(str(value))

        return identifiers

    @staticmethod
    def _has_qualifier(value: object) -> bool:
        """Return whether a qualifier is non-null and non-empty."""
        if isinstance(value, list):
            return len(value) > 0

        return bool(pd.notna(value))

    def lookup(
        self,
        symbol: str,
        symbol_category: SymbolCategory | SymbolCategoryOption = SymbolCategory.PRIMARY,
        match_type: MatchType = MatchType.IDENTICAL,
        search_column: SearchColumn = SearchColumn.BOTH,
    ) -> pd.DataFrame:
        """Lookup a gene symbol in one specific relationship category."""
        symbol_category = SymbolCategory(symbol_category)
        match_type = MatchType(match_type)
        search_column = SearchColumn(search_column)

        df = self.dfs[symbol_category]
        target = symbol.casefold()

        symbol_column = (
            "gene_symbol"
            if symbol_category == SymbolCategory.PRIMARY
            else "alias_symbol"
        )

        symbols = df[symbol_column].astype("string").str.casefold()
        primary_symbols = df["primary_gene_symbol"].astype("string").str.casefold()

        if match_type is MatchType.IDENTICAL:
            symbol_mask = symbols.eq(target).fillna(False)
            primary_mask = primary_symbols.eq(target).fillna(False)
        else:
            symbol_mask = symbols.str.contains(
                target,
                regex=False,
                na=False,
            )
            primary_mask = primary_symbols.str.contains(
                target,
                regex=False,
                na=False,
            )

        if search_column is SearchColumn.ALIAS:
            mask = symbol_mask
        elif search_column is SearchColumn.PRIMARY:
            mask = primary_mask
        else:
            mask = symbol_mask | primary_mask

        return df.loc[mask].copy()

    def resolve(
        self,
        symbol: str,
        match_type: MatchType = MatchType.IDENTICAL,
        *,
        context: ResolutionContext | None = None,
        strict_context: SymbolCategory | StrictContextOption | None = None,
    ) -> pd.DataFrame | str:
        """Resolve a symbol into ranked candidate genes.

        ``context`` changes rank priorities across all relationships.
        ``context`` and ``strict_context`` cannot be supplied together.

        ``strict_context`` restricts the search to a category or named group.

        If a strict search has no matches, return a descriptive string.
        Otherwise, no matches return an empty DataFrame.
        """
        match_type = MatchType(match_type)

        if context is not None and strict_context is not None:
            msg = "context and strict_context are mutually exclusive; provide only one."
            raise ValueError(msg)

        if context is None or context == "default":
            rank_map = self.rank_map
        else:
            try:
                rank_order = self.CONTEXT_RANK_ORDERS[context]
            except (KeyError, TypeError):
                msg = f"Invalid context: {context!r}"
                raise ValueError(msg) from None

            rank_map = {category: rank for rank, category in enumerate(rank_order)}

        if strict_context is None:
            search_categories = set(self.dfs)

        elif (
            isinstance(strict_context, str)
            and strict_context in self.STRICT_CONTEXT_GROUPS
        ):
            search_categories = self.STRICT_CONTEXT_GROUPS[strict_context]

        else:
            try:
                search_categories = {SymbolCategory(strict_context)}
            except (ValueError, TypeError):
                msg = f"Invalid strict_context: {strict_context!r}"
                raise ValueError(msg) from None

        target = symbol.casefold()
        matches: list[pd.DataFrame] = []

        for category, df in self.dfs.items():
            if category not in search_categories:
                continue

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

            category_matches["matched_category"] = category.value
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
            if strict_context is not None:
                return f'No {strict_context} matching "{symbol}"'

            return pd.DataFrame(
                columns=[
                    "candidate_rank",
                    "primary_gene_symbol",
                    "HGNC_ID",
                    "NCBI_ID",
                    "ENSG_ID",
                    "best_category",
                    "qualifier",
                    "matched_symbol",
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
                key=lambda category: rank_map[SymbolCategory(category)],
            )

            best_category = matched_categories[0]

            best_category_rows = group[group["matched_category"] == best_category]

            qualifier: dict[str, object] = {}

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
                    "_category_rank": rank_map[SymbolCategory(best_category)],
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

    def _flatten_unique(self, values: pd.Series) -> list[object]:
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
