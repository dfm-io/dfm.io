import os
import glob

for filename in sorted(glob.glob("posts/*.ipynb")):
    notebook = os.path.splitext(os.path.split(filename)[-1])[0]
    if os.path.exists("workflow/envs/%s.yml" % notebook):
        rule:
            input:
                "posts/%s.ipynb" % notebook
            output:
                "results/posts/%s.ipynb" % notebook
            conda:
                "../envs/%s.yml" % notebook
            log:
                "results/logs/%s.log" % notebook
            shell:
                "jupyter nbconvert --to notebook --output=../{output} --execute {input} &> {log}"

    else:
        rule:
            input:
                "posts/%s.ipynb" % notebook
            output:
                "results/posts/%s.ipynb" % notebook
            conda:
                "../envs/execute.yml"
            log:
                "results/logs/%s.log" % notebook
            shell:
                "jupyter nbconvert --to notebook --output=../{output} --execute {input} &> {log}"
