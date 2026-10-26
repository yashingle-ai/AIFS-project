"""End-to-end test: verifies that financial statement tables are correctly
reconstructed and that the enrichment pipeline produces accurate
row-column-value facts.

NOTE: Uses ASCII markers instead of emoji for Windows console compatibility.

Run from the rag_app directory:
    python _test_table_fix.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import fitz
from pathlib import Path
from src.table_reconstruct import reconstruct_financial_table, is_financial_results_page
from src.enrichment import extract_table_relationships

PDF_PATH = Path("storage/Sample_-_1-88da90ce4785/source.pdf")

def test_page(pg_idx: int, expected: dict, page_name: str):
    """Test a single page's reconstruction and enrichment.
    
    expected = {
        "row_label_fragment": {"col_header_fragment": "expected_value", ...},
        ...
    }
    """
    doc = fitz.open(str(PDF_PATH))
    page = doc[pg_idx]
    page_dict = page.get_text("dict", sort=True)
    page_text = page.get_text("text")
    doc.close()

    assert is_financial_results_page(page_text), \
        f"FAIL: Page {pg_idx+1} not detected as financial results"

    md_table = reconstruct_financial_table(page_dict, page_text)
    assert md_table is not None, \
        f"FAIL: Could not reconstruct table on page {pg_idx+1}"
    assert len(md_table) > 200, \
        f"FAIL: Reconstructed table too short ({len(md_table)} chars)"

    # Run enrichment
    metrics, relationships = extract_table_relationships(md_table)
    assert len(relationships) > 5, \
        f"FAIL: Too few relationships extracted ({len(relationships)})"

    print(f"\n{'='*70}")
    print(f"  {page_name} (Page {pg_idx+1})")
    print(f"{'='*70}")
    print(f"  Table: {len(md_table)} chars, {len(md_table.splitlines())} rows")
    print(f"  Relationships: {len(relationships)}")
    print(f"  Metrics: {len(metrics)}")

    # Verify each expected value
    all_passed = True
    for row_frag, col_vals in expected.items():
        for col_frag, expected_val in col_vals.items():
            # Find matching relationship
            found = False
            for r in relationships:
                if row_frag.lower() in r.lower() and expected_val in r:
                    # Also check column header is present
                    if col_frag.lower() in r.lower():
                        found = True
                        print(f"  [PASS] {row_frag} | {col_frag} = {expected_val}")
                        break
            if not found:
                print(f"  [FAIL] {row_frag} | {col_frag} = {expected_val} — NOT FOUND")
                # Show what we DID find for this row
                matching = [r for r in relationships if row_frag.lower() in r.lower()]
                if matching:
                    print(f"     Found instead: {matching[:3]}")
                all_passed = False

    # Negative tests: verify values are NOT misattributed
    return all_passed


def main():
    if not PDF_PATH.exists():
        print(f"PDF not found: {PDF_PATH}")
        print("Please ensure the Sample document is indexed.")
        sys.exit(1)

    results = []

    # ---- Test 1: Page 18 — Standalone Financial Results ----
    results.append(test_page(17, {
        "Finance cost": {
            "30-09-2025": "469.80",
            "30-06-2025": "349.63",
            "30-09-2024": "206.44",
        },
        "Other Income": {
            "30-09-2025": "542.85",
            "30-06-2025": "492.38",
        },
        "Revenue from Operations": {
            "30-09-2025": "5,066.43",
        },
        "Total Income": {
            "30-09-2025": "5,609.28",
        },
        "Total Expenses": {
            "30-09-2025": "4,912.86",
        },
        "Profit": {  # Profit before exceptional items
            "30-09-2025": "696.42",
        },
    }, "Standalone P&L"))

    # ---- Negative test: 542.85 must NOT be under Finance cost ----
    doc = fitz.open(str(PDF_PATH))
    page = doc[17]
    md = reconstruct_financial_table(page.get_text("dict", sort=True), page.get_text("text"))
    doc.close()
    _, rels = extract_table_relationships(md)
    bad = [r for r in rels if "Finance cost" in r and "542.85" in r]
    if bad:
        print(f"\n  [FAIL] NEGATIVE TEST FAILED: 542.85 wrongly attributed to Finance cost: {bad}")
        results.append(False)
    else:
        print(f"\n  [PASS] NEGATIVE: 542.85 is NOT attributed to Finance cost")
        results.append(True)

    # ---- Test 2: Page 9 — Consolidated Financial Results ----
    results.append(test_page(8, {
        "Interest and Other finance cost": {
            "30-09-2025": "1,761.64",
            "30-06-2025": "1,511.11",
        },
        "Other Income": {
            "30-09-2025": "595.56",
        },
        "Revenue from Operations": {
            "30-09-2025": "21,248.51",
        },
        "Profit": {  # Profit/(Loss) after tax from Continuing Operations
            "30-09-2025": "4,397.63",  # before tax
        },
        "after tax from Continuing": {
            "30-09-2025": "3,414.43",
            "30-06-2025": "976.48",
            "30-09-2024": "1,994.73",
        },
    }, "Consolidated P&L"))

    # ---- Summary ----
    print(f"\n{'='*70}")
    if all(results):
        print("  [PASS] ALL TESTS PASSED — table reconstruction is correct!")
    else:
        print("  [FAIL] SOME TESTS FAILED — see above for details")
        sys.exit(1)


if __name__ == "__main__":
    main()
