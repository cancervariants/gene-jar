"""Tests for the GeneJar resolver."""

import pandas as pd
import pytest

from gene_jar import GeneJar
from gene_jar.resolver import MatchType, SearchColumn, SymbolCategory


# Reusable pytest fixtures to avoid manually initializing all 14 dfs for each test
## gene_jar_factory provides a reusable way to construct GeneJar instances with whichever relationship datasets a particular test needs
## primary_df provides reusable example gene records
@pytest.fixture
def gene_jar_factory():
    """Create a GeneJar with optional primary and alias data."""

    def create(primary_df=None, **category_dfs):
        if primary_df is None:
            primary_df = pd.DataFrame(
                columns=[
                    "gene_symbol",
                    "primary_gene_symbol",
                    "HGNC_ID",
                    "NCBI_ID",
                    "ENSG_ID",
                ]
            )

        def empty_alias_df():
            return pd.DataFrame(
                columns=[
                    "alias_symbol",
                    "primary_gene_symbol",
                    "HGNC_ID",
                    "NCBI_ID",
                    "ENSG_ID",
                ]
            )

        arguments = {
            "primary_df": primary_df,
            "ortholog_df": empty_alias_df(),
            "flj_clone_df": empty_alias_df(),
            "phenotype_df": empty_alias_df(),
            "hgnc_gene_group_df": empty_alias_df(),
            "gene_id_df": empty_alias_df(),
            "mgi_withdrawn_df": empty_alias_df(),
            "ncbi_related_gene_df": empty_alias_df(),
            "ncbi_gene_interaction_df": empty_alias_df(),
            "ncbi_gene_neighbor_df": empty_alias_df(),
            "placeholder_df": empty_alias_df(),
            "previous_df": empty_alias_df(),
            "protein_mass_df": empty_alias_df(),
            "alternate_abbreviation_df": empty_alias_df(),
        }

        arguments.update(category_dfs)
        return GeneJar(**arguments)

    return create


@pytest.fixture
def primary_df():
    """Provide primary gene records with identifiers."""
    return pd.DataFrame(
        {
            "gene_symbol": ["A1BG", "TP53", "KRAS", "NRAS"],
            "primary_gene_symbol": ["A1BG", "TP53", "KRAS", "NRAS"],
            "HGNC_ID": ["HGNC:5", "HGNC:11998", "HGNC:6407", "HGNC:7989"],
            "NCBI_ID": ["1", "7157", "3845", "4893"],
            "ENSG_ID": [
                "ENSG00000121410",
                "ENSG00000141510",
                "ENSG00000133703",
                "ENSG00000213281",
            ],
        }
    )


# Primary gene symbol resolution


def test_resolve_primary_gene_symbol(gene_jar_factory, primary_df):
    """Test that a primary symbol resolves to its gene.

    :param gene_jar_factory: Fixture for creating a GeneJar instance.
    :param primary_df: DataFrame containing primary gene records.
    """
    gj = gene_jar_factory(primary_df=primary_df)

    result = gj.resolve("A1BG")

    assert len(result) == 1
    assert result.iloc[0]["primary_gene_symbol"] == "A1BG"
    assert result.iloc[0]["best_category"] == SymbolCategory.PRIMARY
    assert result.iloc[0]["candidate_rank"] == 1
    assert result.iloc[0]["relationship_count"] == 1


def test_resolve_primary_gene_identifiers(gene_jar_factory, primary_df):
    """Test that gene identifiers are returned as sets."""
    gj = gene_jar_factory(primary_df=primary_df)

    result = gj.resolve("TP53")
    candidate = result.iloc[0]

    assert candidate["HGNC_ID"] == {"HGNC:11998"}
    assert candidate["NCBI_ID"] == {"7157"}
    assert candidate["ENSG_ID"] == {"ENSG00000141510"}


def test_resolve_is_case_insensitive(gene_jar_factory, primary_df):
    """Test that symbol matching ignores capitalization."""
    gj = gene_jar_factory(primary_df=primary_df)

    result = gj.resolve("tp53")

    assert len(result) == 1
    assert result.iloc[0]["primary_gene_symbol"] == "TP53"


def test_resolve_unknown_symbol(gene_jar_factory, primary_df):
    """Test that unknown symbols return an empty DataFrame."""
    gj = gene_jar_factory(primary_df=primary_df)

    result = gj.resolve("UNKNOWN_SYMBOL")

    assert result.empty
    assert "candidate_rank" in result.columns
    assert "primary_gene_symbol" in result.columns
    assert "best_category" in result.columns


# Alias resolution


def test_resolve_previous_symbol(gene_jar_factory):
    """Test that a previous symbol resolves to its primary gene."""
    previous_df = pd.DataFrame(
        {
            "alias_symbol": ["OLDTP53"],
            "primary_gene_symbol": ["TP53"],
            "Previous Symbol Source": ["HGNC"],
        }
    )

    gj = gene_jar_factory(previous_df=previous_df)

    result = gj.resolve("OLDTP53")
    candidate = result.iloc[0]

    assert candidate["primary_gene_symbol"] == "TP53"
    assert candidate["best_category"] == SymbolCategory.PREVIOUS
    assert candidate["qualifier"] == {"Previous Symbol Source": "HGNC"}

    @pytest.mark.parametrize(
        ("matching_species", "expected_qualifier"),
        [
            (
                ["mouse", "rat"],
                {"Matching Species": ["mouse", "rat"]},
            ),
            (
                [],
                {},
            ),
            (
                None,
                {},
            ),
        ],
    )
    def test_resolve_ortholog_symbol(
        gene_jar_factory,
        matching_species,
        expected_qualifier,
    ):
        """Test ortholog resolution with populated and absent qualifiers."""
        ortholog_df = pd.DataFrame(
            {
                "alias_symbol": ["TEST_ORTHOLOG"],
                "primary_gene_symbol": ["TP53"],
                "Matching Species": [matching_species],
            }
        )

        gj = gene_jar_factory(ortholog_df=ortholog_df)

        result = gj.resolve("TEST_ORTHOLOG")
        candidate = result.iloc[0]

        assert candidate["primary_gene_symbol"] == "TP53"
        assert candidate["best_category"] == SymbolCategory.ORTHOLOG
        assert candidate["qualifier"] == expected_qualifier


def test_resolve_multiple_relationship_categories(gene_jar_factory):
    """Test that relationships for the same gene are combined."""
    previous_df = pd.DataFrame(
        {
            "alias_symbol": ["OLDGENE"],
            "primary_gene_symbol": ["TP53"],
            "Previous Symbol Source": ["HGNC"],
        }
    )

    ortholog_df = pd.DataFrame(
        {
            "alias_symbol": ["OLDGENE"],
            "primary_gene_symbol": ["TP53"],
            "Matching Species": [["mouse"]],
        }
    )

    gj = gene_jar_factory(
        previous_df=previous_df,
        ortholog_df=ortholog_df,
    )

    result = gj.resolve("OLDGENE")
    candidate = result.iloc[0]

    assert len(result) == 1
    assert candidate["relationship_count"] == 2
    assert candidate["matched_categories"] == [
        SymbolCategory.PREVIOUS,
        SymbolCategory.ORTHOLOG,
    ]
    assert candidate["best_category"] == SymbolCategory.PREVIOUS


# Candidate ranking


def test_resolve_prioritizes_primary_over_alias(gene_jar_factory):
    """Test that primary matches rank above alias matches."""
    primary_df = pd.DataFrame(
        {
            "gene_symbol": ["TEST"],
            "primary_gene_symbol": ["TEST"],
        }
    )

    previous_df = pd.DataFrame(
        {
            "alias_symbol": ["TEST"],
            "primary_gene_symbol": ["OTHER"],
            "Previous Symbol Source": ["HGNC"],
        }
    )

    gj = gene_jar_factory(
        primary_df=primary_df,
        previous_df=previous_df,
    )

    result = gj.resolve("TEST")

    assert result["primary_gene_symbol"].tolist() == ["TEST", "OTHER"]
    assert result["candidate_rank"].tolist() == [1, 2]


def test_resolve_prioritizes_previous_over_ortholog(gene_jar_factory):
    """Test that category priority determines candidate ranking."""
    previous_df = pd.DataFrame(
        {
            "alias_symbol": ["SHARED"],
            "primary_gene_symbol": ["GENE_A"],
            "Previous Symbol Source": ["HGNC"],
        }
    )

    ortholog_df = pd.DataFrame(
        {
            "alias_symbol": ["SHARED"],
            "primary_gene_symbol": ["GENE_B"],
            "Matching Species": [["mouse"]],
        }
    )

    gj = gene_jar_factory(
        previous_df=previous_df,
        ortholog_df=ortholog_df,
    )

    result = gj.resolve("SHARED")

    assert result["primary_gene_symbol"].tolist() == ["GENE_A", "GENE_B"]


def test_resolve_breaks_ties_by_relationship_count(gene_jar_factory):
    """Test that more matched categories break equal-priority ties."""
    previous_df = pd.DataFrame(
        {
            "alias_symbol": ["SHARED", "SHARED"],
            "primary_gene_symbol": ["GENE_A", "GENE_B"],
            "Previous Symbol Source": ["HGNC", "HGNC"],
        }
    )

    ortholog_df = pd.DataFrame(
        {
            "alias_symbol": ["SHARED"],
            "primary_gene_symbol": ["GENE_B"],
            "Matching Species": [["mouse"]],
        }
    )

    gj = gene_jar_factory(
        previous_df=previous_df,
        ortholog_df=ortholog_df,
    )

    result = gj.resolve("SHARED")

    assert result["primary_gene_symbol"].tolist() == ["GENE_B", "GENE_A"]
    assert result["relationship_count"].tolist() == [2, 1]


def test_resolve_breaks_remaining_ties_alphabetically(gene_jar_factory):
    """Test alphabetical ranking when category and count are equal."""
    previous_df = pd.DataFrame(
        {
            "alias_symbol": ["SHARED", "SHARED"],
            "primary_gene_symbol": ["GENE_B", "GENE_A"],
            "Previous Symbol Source": ["HGNC", "HGNC"],
        }
    )

    gj = gene_jar_factory(previous_df=previous_df)

    result = gj.resolve("SHARED")

    assert result["primary_gene_symbol"].tolist() == ["GENE_A", "GENE_B"]


# Match types


def test_resolve_identical_match(gene_jar_factory, primary_df):
    """Test that identical matching excludes substring matches."""
    gj = gene_jar_factory(primary_df=primary_df)

    result = gj.resolve("RAS", match_type=MatchType.IDENTICAL)

    assert result.empty


def test_resolve_partial_match(gene_jar_factory, primary_df):
    """Test that partial matching finds symbols containing a substring."""
    gj = gene_jar_factory(primary_df=primary_df)

    result = gj.resolve("RAS", match_type=MatchType.PARTIAL)

    assert set(result["primary_gene_symbol"]) == {"KRAS", "NRAS"}


def test_resolve_invalid_match_type(gene_jar_factory, primary_df):
    """Test that invalid match types raise ValueError."""
    gj = gene_jar_factory(primary_df=primary_df)

    with pytest.raises(ValueError, match="'invalid' is not a valid MatchType"):
        gj.resolve("TP53", match_type="invalid")


# Lookup


@pytest.mark.parametrize(
    ("search_column", "expected_symbols"),
    [
        (SearchColumn.ALIAS, ["GENE_A"]),
        (SearchColumn.PRIMARY, ["SHARED"]),
        (SearchColumn.BOTH, ["GENE_A", "SHARED"]),
    ],
)
def test_lookup_search_columns(
    gene_jar_factory,
    search_column,
    expected_symbols,
):
    """Test lookup using alias, primary, or both columns."""
    previous_df = pd.DataFrame(
        {
            "alias_symbol": ["SHARED", "OTHER"],
            "primary_gene_symbol": ["GENE_A", "SHARED"],
        }
    )

    gj = gene_jar_factory(previous_df=previous_df)

    result = gj.lookup(
        "SHARED",
        symbol_category=SymbolCategory.PREVIOUS,
        search_column=search_column,
    )

    assert result["primary_gene_symbol"].tolist() == expected_symbols


def test_lookup_partial_match(gene_jar_factory, primary_df):
    """Test partial matching through lookup."""
    gj = gene_jar_factory(primary_df=primary_df)

    result = gj.lookup("RAS", match_type=MatchType.PARTIAL)

    assert set(result["primary_gene_symbol"]) == {"KRAS", "NRAS"}


@pytest.mark.parametrize(
    ("argument", "value", "expected_error"),
    [
        ("symbol_category", "invalid", "'invalid' is not a valid SymbolCategory"),
        ("match_type", "invalid", "'invalid' is not a valid MatchType"),
        ("search_column", "invalid", "'invalid' is not a valid SearchColumn"),
    ],
)
def test_lookup_invalid_options(gene_jar_factory, argument, value, expected_error):
    """Test that invalid lookup options raise ValueError."""
    gj = gene_jar_factory()

    with pytest.raises(ValueError, match=expected_error):
        gj.lookup("TP53", **{argument: value})


# Ambiguity detection


def test_ambiguity_check_unambiguous(gene_jar_factory, primary_df):
    """Test that one matching gene is not ambiguous."""
    gj = gene_jar_factory(primary_df=primary_df)

    result = gj.ambiguity_check("TP53")

    assert result.is_ambiguous is False
    assert len(result.gene_matches) == 1
    assert result.gene_matches[0].primary_gene_symbol == "TP53"


def test_ambiguity_check_ambiguous(gene_jar_factory):
    """Test that a symbol associated with two genes is ambiguous."""
    previous_df = pd.DataFrame(
        {
            "alias_symbol": ["SHARED", "SHARED"],
            "primary_gene_symbol": ["GENE_A", "GENE_B"],
            "HGNC_ID": ["HGNC:1", "HGNC:2"],
            "Previous Symbol Source": ["HGNC", "HGNC"],
        }
    )

    gj = gene_jar_factory(previous_df=previous_df)

    result = gj.ambiguity_check("SHARED")

    assert result.is_ambiguous is True
    assert len(result.gene_matches) == 2
    assert {gene.primary_gene_symbol for gene in result.gene_matches} == {
        "GENE_A",
        "GENE_B",
    }


def test_ambiguity_check_unknown_symbol(gene_jar_factory):
    """Test that an unknown symbol has no gene matches."""
    gj = gene_jar_factory()

    result = gj.ambiguity_check("UNKNOWN")

    assert result.is_ambiguous is False
    assert result.gene_matches == []
