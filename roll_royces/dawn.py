# Dawn formatter - Used to copy to clipboard the result

import re
import os
import pyperclip
import pandas as pd

QUERY_CONFIG = {
    "directory_format": "{_}_{dataset}_{missing_percentage}_{seed}_{combination_mode}",
    "where": {
        "dataset": "PhuLien",
        "seed": 7645,
        "missing_percentage": 6,
        "combination_mode": "meow"
    },
    "col": {
        "target": "Model",
        "only": ["BaTriTempCNN1D"]
    },
    "row": {
        "only": ["Similarity", "NMAE", "RMSE", "R2", "FSD", "FB", "FA2"]
    },
    "out_delimiter": "\t",
}


if __name__ == "__main__":
    # Get the query param from directory format
    query = QUERY_CONFIG["directory_format"]
    query_keys = re.findall(r"\{([^}]+)\}", query)

    # Build regex pattern
    pattern = rf"^{query}$"
    for key in query_keys:
        pattern = pattern.replace(f"{{{key}}}", f"(.+?)")

    # List all directories in results
    directories = os.listdir("results")

    # Get the where clause
    where = QUERY_CONFIG["where"]

    # Get the col clause
    col = QUERY_CONFIG["col"]
    col_target = col.get("target", "Model")
    col_only = col.get("only", [])

    # Get the row clause
    row = QUERY_CONFIG["row"]
    row_only = row.get("only", [])

    # Get the out delimiter
    out_delimiter = QUERY_CONFIG["out_delimiter"]

    # Find the directories that match the where clause
    query_directories = []
    for directory in directories:
        _match = re.match(pattern, directory)
        if _match:
            match_tuple = _match.groups()
            # Check if the where clause is satisfied
            if all(str(match_tuple[query_keys.index(key)]) == str(value) for key, value in where.items()):
                query_directories.append(directory)

    # Iterate over the query directories
    results = []
    for directory in query_directories:
        # Read the result file
        result_file = os.path.join(
            os.getcwd(), "results", directory, "metrics.csv")
        if os.path.exists(result_file):
            df = pd.read_csv(result_file)
            df = df[df[col_target].isin(col_only)]

            # Remove the col_target
            df = df.drop(columns=[col_target])

            # Filter the rows
            df = df.loc[:, row_only]

            # Format the results
            for r in df.values:
                results.append(f"{out_delimiter}".join(map(str, r)))

    # Copy the results to clipboard
    results = "\n".join(results)
    print("📋", results)
    pyperclip.copy(results)
