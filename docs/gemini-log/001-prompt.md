prompt = """
You are an R project initializer. In this repository, output ONLY shell commands and R console commands 
(each prefixed with $ for shell or > for R) to:
1) create an R package skeleton (name: fedavgR),
2) initialize renv,
3) add testthat + roxygen2 + devtools,
4) create DESCRIPTION (Title: "Federating Learning with FedAvg in R", License: MIT + file LICENSE),
5) create a .gitignore for R + renv + docs,
6) create README.md with sections: Overview, Install, Quickstart, Reproducibility, GenAI Usage Instructions.

Constraints:
- Do not execute anything yourself—just write the commands.
- Keep the output as a sequence of commands I can copy/paste.
"""
