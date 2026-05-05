# agent rules

## priorities
- optimize for simplicity, boringness, and long-term maintainability
- prefer minimal diffs; avoid refactors unless required for the active task

## tech stack
- python; scrapy
- file storage: json or csv
- assume local virtual env is available and accessible
- do not add new dependencies unless explicitly approved; if unavoidable, document justification in the active task notes

## workflow
- prefer direct argv commands (no bash -lc / compound shell chains) unless necessary
- work on ONE task at a time unless explicitly instructed otherwise
- at the start of work, state the task id you are executing
- do not start work unless a task id is specified; if missing, choose the earliest unchecked task and say so
- propose incremental steps
- always include basic tests for core logic
- when you complete a task:
  - mark it [X] in docs/tasks.md
  - fill in evidence with commit hash + commands run
  - never mark complete unless acceptance criteria are met
  - include date and time (HH:MM)

```
* [ ] t1.1 Task Title (1)
Description and PM notes
** acceptance criteria
1. AC 1
2. AC 2

** notes
- document thoughts, decisions, reasoning

** evidence
- commit: 
- tests: 
- datetime: 
```
