#!/usr/bin/env bash
tar czf data/data.tgz manifest.md data/documents
tar czf data_processed.tgz data

git ls-tree --full-tree -r --name-only HEAD | xargs tar cvf final_project.tar
tar rvf final_project.tar final_report.pdf
gzip -9 final_project.tar